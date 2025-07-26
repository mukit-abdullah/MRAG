# Hey there! Welcome to My Multilingual RAG System

So, I built this cool little system that can read Bengali PDFs and answer your questions about them - whether you ask in Bengali or English! Think of it as having a smart friend who's really good at reading Bengali documents and can chat with you about what they contain.


### Step 1: Get the Right Tools
First things first, you'll need to install a few Python packages. Don't panic - it's just one command:

```bash
pip install sentence-transformers faiss-cpu pdfplumber
```

### Step 2: Get Your PDF Ready
- Drop your Bengali PDF into the same folder as the code
- I've set it up to look for `HSC26-Bangla1st-Paper2.pdf` by default
- Want to use a different file? Just change the filename in the code - it's easy to find!

### Step 3: Fire It Up!
```bash
python rag_system.py
```

### Step 4: Start Chatting!
- The first time you run it, give it a moment to read through your PDF (it's doing some smart stuff behind the scenes)
- Then just start asking questions! 
- When you're done, just type 'exit' and you're good to go



### The Tools I'm Using

**The Heavy Lifters:**
- **sentence-transformers** - This is the brain that understands what your text actually *means* (not just keywords!)
- **faiss-cpu** - Super fast at finding similar stuff (like Google search, but for document chunks)
- **pdfplumber** - The PDF reader that actually gets Bengali text right

**The Supporting Cast:**
- **os, re, json, logging, datetime** - Basic Python tools for file handling, text cleaning, and keeping track of things
- **typing** - Just makes the code cleaner (I'm a bit of a perfectionist)

**The AI Magic:**
- **SentenceTransformer** - Creates these 768-number "fingerprints" that capture what text means
- **FAISS** - Lightning-fast similarity search (seriously, it's impressive)
- **Smart text patterns** - Custom regex that understands both Bengali and English


### Bengali Questions (বাংলা প্রশ্ন)

**Want to know someone's age?**
```
You: বিয়ের সময় কলযাণীর প্রকৃত বয়স কত ছিল?
System: ষোল বছর।
(Found this info from 3 different parts of the document)
```

**Curious about characters?**
```
You: এই গল্পের মূল চরিত্র কে?
System: কলযাণী এই গল্পের মূল চরিত্র।
(Checked 4 sections to give you this answer)
```

**Want story details?**
```
You: কলযাণীর বিয়ে কেন হয়েছিল?
System: পারিবারিক সিদ্ধান্ত অনুযায়ী কলযাণীর বিয়ে হয়েছিল।
(Found relevant info in 2 document sections)
```

### English Questions

**General stuff:**
```
You: What is the main topic of this document?
System: This document discusses the story of Kalyani and her marriage circumstances.
(Looked through 5 sections to understand the overall theme)
```

**Character analysis:**
```
You: Who is the protagonist in this story?
System: Kalyani is the main protagonist of this story.
(Cross-referenced 3 sections to confirm)
```

**Specific details:**
```
You: What was Kalyani's age at marriage?
System: Sixteen years old.
(Found this in 2 relevant sections)
```

### Mix It Up! (English + Bengali)

```
You: How old was কলযাণী when she got married?
System: ষোল বছর বয়সে কলযাণীর বিয়ে হয়েছিল।
(The system is smart enough to respond in the appropriate language!)
```

## How Does This Magic Actually Work?

Let me break down what happens when you ask a question:

1. **PDF Reading** - First, I extract all the text from your PDF (Bengali Unicode and all!)
2. **Smart Chunking** - I break the text into meaningful pieces (about 500 characters each, with some overlap so nothing gets lost)
3. **AI Understanding** - Each chunk gets converted into a 768-number "meaning fingerprint"
4. **Your Question** - When you ask something, your question gets the same treatment
5. **Smart Matching** - The system finds the chunks that are most similar to your question
6. **Answer Generation** - Finally, it crafts an answer based on the most relevant information it found

## The Technical Stuff (For the Curious)





Just run this and you'll be fine:
```bash
pip install sentence-transformers faiss-cpu pdfplumber
```

**"Can't find the PDF file"**
Make sure your PDF is in the same folder as the Python script, or update the filename in the code.

**"The answers seem weird"**
Try rephrasing your question, or check if that information is actually in your PDF. Sometimes the AI gets confused if the question is too vague.

**"It's using too much memory"**
If you're working with a huge document, you might need to use a smaller PDF or adjust the chunk size in the code.

**Want to see what's happening behind the scenes?**
Add this to the top of the script:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## The Story Behind the Choices (Why I Built It This Way)

### Why pdfplumber for text extraction?

I tried a bunch of different approaches, and here's what I learned:
- **pdfplumber just works better with Bengali** - Other libraries would give me garbled text or miss characters
- **No extra software needed** - Unlike Tesseract OCR, you don't need to install anything else on your computer
- **Handles mixed content well** - When you have Bengali and English in the same document, it keeps everything straight

**The challenges I faced:**
- Bengali text would sometimes come out as weird symbols (fixed with proper UTF-8 handling)
- Line breaks were all over the place (solved with some smart text cleaning)
- Pages would break in the middle of sentences (added special handling for page boundaries)

### Why this chunking approach?

I experimented with different ways to break up the text:
- **Sentence-based splitting** - This preserves meaning much better than just cutting at random character counts
- **500 characters with 50-character overlap** - Sweet spot between having enough context and keeping chunks focused
- **Bengali-aware** - Uses Bengali punctuation (।) properly, not just English periods

This approach works because:
- Each chunk contains complete thoughts
- The overlap means important info doesn't get lost at boundaries
- It's the right size for the AI model to understand fully

### Why the multilingual-e5-base model?

I could have used other embedding models, but this one is special:
- **Actually trained on Bengali** - Not just English with Bengali thrown in as an afterthought
- **Semantic understanding** - It gets what text *means*, not just what words are there
- **Good balance** - Fast enough for real-time use, smart enough to be accurate
- **Cross-language magic** - Maps Bengali and English to the same "meaning space"

### Why FAISS for search?

- **Speed** - It's ridiculously fast at finding similar vectors
- **Accuracy** - With normalized embeddings, inner product = cosine similarity (the gold standard)
- **Scalable** - Can handle way more documents than I'm using without breaking a sweat
- **Memory efficient** - Doesn't eat up all your RAM

### How I handle tricky queries

**When your question is vague:**
- The system tries to find broadly relevant content
- If it can't find anything good, it honestly tells you "I don't have enough info"
- It combines multiple document sections to give you a fuller picture

**When the document doesn't have the answer:**
- No hallucination! It won't make stuff up
- Clear "insufficient information" messages
- Suggests rephrasing or checking if the info exists in your document

### What could make this even better?

**Better chunking:**
- Could use topic modeling to create more coherent chunks
- Dynamic overlap based on content type
- Different chunk sizes for different kinds of questions

**Smarter models:**
- Fine-tuned specifically on Bengali literature
- Larger models with more parameters (but slower)
- Newer architectures as they come out

**Better documents:**
- Higher quality OCR for scanned documents
- Multiple document support for cross-referencing
- Better handling of tables, images, and structured content

**System improvements:**
- Automatic query expansion with synonyms
- Learning from user feedback
- Better handling of complex, multi-part questions

## What's Next?

I'm thinking about adding:
- Support for more powerful language models (GPT, Claude, etc.)
- Web interface so you don't need to use the command line
- Support for Word documents and plain text files
- Even better Bengali text processing
- Ability to work with multiple documents at once

## Final Thoughts

This has been a fun project to build! The goal was to create something that actually works well with Bengali content - not just English with Bengali as an afterthought. I hope it's useful for anyone working with Bengali documents who wants a smart way to search and ask questions about their content.

Feel free to experiment with it, break it, improve it, or just use it as-is. If you run into issues or have ideas for making it better, I'd love to hear about them!


