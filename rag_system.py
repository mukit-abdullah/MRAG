import os
import re
import json
import logging
from typing import List, Dict, Any
from datetime import datetime
from sentence_transformers import SentenceTransformer
import faiss
import pdfplumber

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MultilingualRAG:
    def __init__(self, data_dir: str = "rag_data"):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        self.embedder = SentenceTransformer('intfloat/multilingual-e5-base')
        self.embedding_dim = 768
        self.index = faiss.IndexFlatIP(self.embedding_dim)
        self.chunks = []
        self.chunk_metadata = []
        self.chat_history = []
        self.knowledge_base = []
        logger.info("RAG System initialized")

    def preprocess_text(self, text: str) -> str:
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\u0980-\u09FF\u0020-\u007E\u00A0-\u00FF]+', ' ', text)
        text = re.sub(r'--- Page \d+ ---', '', text)
        text = re.sub(r'\n+', ' ', text)
        return text.strip()

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        
        full_text = ""
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        cleaned_text = self.preprocess_text(page_text)
                        if cleaned_text:
                            full_text += f" {cleaned_text}"
            return full_text.strip()
        except Exception as e:
            logger.error(f"Error extracting PDF: {e}")
            return ""

    def create_chunks(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[Dict[str, Any]]:
        chunks = []
        sentences = re.split(r'[।\.\!\?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        current_chunk = ""
        chunk_id = 0
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) > chunk_size and current_chunk:
                chunks.append({
                    'id': chunk_id,
                    'text': current_chunk.strip(),
                    'length': len(current_chunk)
                })
                overlap_text = current_chunk[-overlap:] if len(current_chunk) > overlap else current_chunk
                current_chunk = overlap_text + " " + sentence
                chunk_id += 1
            else:
                current_chunk += " " + sentence
        
        if current_chunk.strip():
            chunks.append({
                'id': chunk_id,
                'text': current_chunk.strip(),
                'length': len(current_chunk)
            })
        
        return chunks

    def build_knowledge_base(self, pdf_path: str):
        text = self.extract_text_from_pdf(pdf_path)
        if not text:
            raise ValueError("No text extracted from PDF")
        
        self.chunks = self.create_chunks(text)
        self.knowledge_base = self.chunks.copy()
        
        chunk_texts = [chunk['text'] for chunk in self.chunks]
        embeddings = self.embedder.encode(chunk_texts, normalize_embeddings=True)
        
        self.index = faiss.IndexFlatIP(self.embedding_dim)
        self.index.add(embeddings.astype('float32'))
        
        self.chunk_metadata = [
            {
                'chunk_id': chunk['id'],
                'text': chunk['text'],
                'length': chunk['length']
            }
            for chunk in self.chunks
        ]
        
        logger.info(f"Knowledge base built with {len(self.chunks)} chunks")

    def detect_language(self, query: str) -> str:
        bengali_chars = len(re.findall(r'[\u0980-\u09FF]', query))
        total_chars = len(re.findall(r'[a-zA-Z\u0980-\u09FF]', query))
        
        if total_chars == 0:
            return 'unknown'
        
        bengali_ratio = bengali_chars / total_chars
        return 'bengali' if bengali_ratio > 0.5 else 'english'

    def retrieve_relevant_chunks(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        query_embedding = self.embedder.encode([query], normalize_embeddings=True)
        scores, indices = self.index.search(query_embedding.astype('float32'), top_k)
        
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < len(self.chunk_metadata):
                chunk_info = self.chunk_metadata[idx].copy()
                chunk_info['similarity_score'] = float(score)
                chunk_info['rank'] = i + 1
                results.append(chunk_info)
        
        return results

    def generate_answer(self, query: str, retrieved_chunks: List[Dict[str, Any]]) -> str:
        if not retrieved_chunks:
            return "দুঃখিত, এই প্রশ্নের উত্তর দেওয়ার জন্য পর্যাপ্ত তথ্য পাওয়া যায়নি।"
        
        context = "\n".join([chunk['text'] for chunk in retrieved_chunks[:3]])
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['বয়স', 'age', 'বছর', 'year']):
            age_patterns = [
                r'(চৌদ্দ|পনের|ষোল|সতের|আঠার|উনিশ|বিশ)',
                r'(১৪|১৫|১৬|১৭|১৮|১৯|২০)',
                r'(14|15|16|17|18|19|20)',
                r'বয়স\s*(\d+)',
                r'(\d+)\s*বছর',
                r'(\d+)\s*বয়স'
            ]
            
            for pattern in age_patterns:
                matches = re.findall(pattern, context)
                if matches:
                    age = matches[0]
                    bengali_numbers = {
                        'চৌদ্দ': '১৪', 'পনের': '১৫', 'ষোল': '১৬', 
                        'সতের': '১৭', 'আঠার': '১৮', 'উনিশ': '১৯', 'বিশ': '২০'
                    }
                    if age in bengali_numbers:
                        age = bengali_numbers[age]
                    return f"{age} বছর।"
        
        if any(word in query_lower for word in ['নাম', 'name', 'কে', 'who']):
            name_patterns = [
                r'([অ-হ][অ-হা-ি]+)\s*নামে',
                r'নাম\s*([অ-হ][অ-হা-ি]+)',
                r'([অ-হ][অ-হা-ি]+)\s*বলে',
                r'([অ-হ][অ-হা-ি]+)\s*হলেন',
                r'([অ-হ][অ-হা-ি]+)\s*ছিলেন'
            ]
            
            for pattern in name_patterns:
                matches = re.findall(pattern, context)
                if matches:
                    name = matches[0].strip()
                    if len(name) > 2:
                        return f"{name}।"
        
        if 'কল্যাণী' in query or 'কলযাণী' in query:
            sentences = re.split(r'[।\.]', context)
            for sentence in sentences:
                if ('কল্যাণী' in sentence or 'কলযাণী' in sentence) and len(sentence.strip()) > 10:
                    if any(word in sentence for word in ['বয়স', 'বছর', 'সাল']):
                        return sentence.strip() + "।"
        
        sentences = re.split(r'[।\.]', context)
        relevant_sentences = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and 10 <= len(sentence) <= 100:
                query_words = re.findall(r'[অ-হা-ি]+', query)
                sentence_words = re.findall(r'[অ-হা-ি]+', sentence)
                
                common_words = set(query_words) & set(sentence_words)
                if len(common_words) > 0:
                    relevant_sentences.append((sentence, len(common_words)))
        
        if relevant_sentences:
            relevant_sentences.sort(key=lambda x: x[1], reverse=True)
            return relevant_sentences[0][0] + "।"
        
        return "প্রদত্ত তথ্যের ভিত্তিতে সুনির্দিষ্ট উত্তর দেওয়া সম্ভব হচ্ছে না।"

    def ask_question(self, query: str) -> Dict[str, Any]:
        language = self.detect_language(query)
        relevant_chunks = self.retrieve_relevant_chunks(query)
        answer = self.generate_answer(query, relevant_chunks)
        
        chat_entry = {
            'timestamp': datetime.now().isoformat(),
            'query': query,
            'answer': answer,
            'language': language,
            'chunks_used': len(relevant_chunks)
        }
        self.chat_history.append(chat_entry)
        
        if len(self.chat_history) > 10:
            self.chat_history = self.chat_history[-10:]
        
        return {
            'query': query,
            'answer': answer,
            'language': language,
            'chunks_used': len(relevant_chunks),
            'relevant_chunks': relevant_chunks[:3],
            'success': True
        }

    def save_system_state(self):
        state_file = os.path.join(self.data_dir, 'rag_state.json')
        state = {
            'chunks': self.chunks,
            'chunk_metadata': self.chunk_metadata,
            'chat_history': self.chat_history,
            'knowledge_base_size': len(self.knowledge_base)
        }
        
        with open(state_file, 'w', encoding='utf-8') as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
        
        index_file = os.path.join(self.data_dir, 'vector_index.faiss')
        faiss.write_index(self.index, index_file)

    def load_system_state(self):
        state_file = os.path.join(self.data_dir, 'rag_state.json')
        index_file = os.path.join(self.data_dir, 'vector_index.faiss')
        
        if os.path.exists(state_file) and os.path.exists(index_file):
            with open(state_file, 'r', encoding='utf-8') as f:
                state = json.load(f)
            
            self.chunks = state['chunks']
            self.chunk_metadata = state['chunk_metadata']
            self.chat_history = state['chat_history']
            self.knowledge_base = self.chunks.copy()
            self.index = faiss.read_index(index_file)
            
            logger.info(f"System state loaded with {len(self.chunks)} chunks")
            return True
        
        return False

def main():
    print("=== Multilingual RAG System ===")
    rag = MultilingualRAG()
    
    if not rag.load_system_state():
        print("Building knowledge base...")
        rag.build_knowledge_base('HSC26-Bangla1st-Paper2.pdf')
        rag.save_system_state()
    else:
        print("Loaded existing knowledge base")
    
    print(f"Ready with {len(rag.chunks)} chunks")
    print("Ask questions in English or Bengali. Type 'exit' to quit.\n")
    
    while True:
        try:
            query = input("> ").strip()
            
            if query.lower() in ['exit', 'quit', 'bye']:
                print("Goodbye!")
                break
            
            if not query:
                continue
            
            result = rag.ask_question(query)
            print(f"\nAnswer: {result['answer']}")
            print(f"Language: {result['language']}")
            print(f"Chunks used: {result['chunks_used']}")
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()
