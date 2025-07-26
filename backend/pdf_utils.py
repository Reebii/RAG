import fitz  # PyMuPDF
import os

def extract_text_from_pdf(pdf_path):
    try:
        print(f"📄 Opening PDF at: {pdf_path}")
        print("📁 Current working dir:", os.getcwd())
        
        # Check if file exists before trying to open
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found at: {pdf_path}")
        
        # Check if file has content
        file_size = os.path.getsize(pdf_path)
        if file_size == 0:
            raise ValueError("PDF file is empty")
        
        print(f"📊 File size: {file_size} bytes")
        
        # Check if file size is suspiciously small (less than 1KB might indicate corruption)
        if file_size < 1024:
            print("⚠️ Warning: File size is very small, might be corrupted")
            
        # Use absolute path to avoid any path resolution issues
        abs_path = os.path.abspath(pdf_path)
        print(f"📂 Absolute path: {abs_path}")
        
        # Try to read first few bytes to check if it's a valid PDF
        with open(abs_path, 'rb') as f:
            header = f.read(10)
            print(f"📝 File header: {header}")
            if not header.startswith(b'%PDF-'):
                raise ValueError("File is not a valid PDF (missing PDF header)")
        
        # Try opening with PyMuPDF
        print("🔓 Attempting to open with PyMuPDF...")
        doc = fitz.open(abs_path)
        
        print(f"📖 PDF opened successfully. Pages: {len(doc)}")
        
        if len(doc) == 0:
            doc.close()
            raise ValueError("PDF has no pages")
        
        text = ""
        
        for page_num, page in enumerate(doc):
            try:
                page_text = page.get_text()
                text += page_text
                print(f"📑 Extracted {len(page_text)} characters from page {page_num + 1}")
            except Exception as page_error:
                print(f"⚠️ Error extracting text from page {page_num + 1}: {page_error}")
                continue
        
        doc.close()
        
        if not text.strip():
            # Try alternative extraction method
            print("🔄 No text found with get_text(), trying alternative method...")
            doc = fitz.open(abs_path)
            for page_num, page in enumerate(doc):
                try:
                    # Try extracting as dictionary (includes more detail)
                    text_dict = page.get_text("dict")
                    for block in text_dict["blocks"]:
                        if "lines" in block:
                            for line in block["lines"]:
                                for span in line["spans"]:
                                    text += span["text"] + " "
                except Exception as alt_error:
                    print(f"⚠️ Alternative extraction failed for page {page_num + 1}: {alt_error}")
            doc.close()
        
        if not text.strip():
            raise ValueError("No text could be extracted from the PDF - it might be an image-based PDF or corrupted")
        
        print(f"✅ Successfully extracted {len(text)} characters total")
        return text
        
    except fitz.FileDataError as e:
        print(f"❌ PyMuPDF FileDataError: {e}")
        print("💡 This usually means the PDF file is corrupted or not a valid PDF")
        raise ValueError(f"Invalid or corrupted PDF file: {e}")
    except fitz.FileNotFoundError as e:
        print(f"❌ PyMuPDF FileNotFoundError: {e}")
        raise FileNotFoundError(f"PDF file not found: {e}")
    except Exception as e:
        print(f"❌ Failed to extract text from PDF: {e}")
        print(f"🔍 Error type: {type(e).__name__}")
        raise

def validate_pdf_file(pdf_path):
    """
    Validate if a file is a proper PDF before processing
    """
    try:
        if not os.path.exists(pdf_path):
            return False, "File does not exist"
        
        file_size = os.path.getsize(pdf_path)
        if file_size == 0:
            return False, "File is empty"
        
        if file_size < 100:
            return False, "File is too small to be a valid PDF"
        
        # Check PDF header
        with open(pdf_path, 'rb') as f:
            header = f.read(10)
            if not header.startswith(b'%PDF-'):
                return False, "File does not have valid PDF header"
        
        # Try to open with PyMuPDF
        try:
            doc = fitz.open(pdf_path)
            page_count = len(doc)
            doc.close()
            
            if page_count == 0:
                return False, "PDF has no pages"
            
            return True, f"Valid PDF with {page_count} pages"
        except Exception as e:
            return False, f"Cannot open PDF: {e}"
    
    except Exception as e:
        return False, f"Validation error: {e}"