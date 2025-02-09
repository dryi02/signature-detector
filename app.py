import cv2
import numpy as np
import sys
import json
import base64
import fitz  # PyMuPDF
import os
from datetime import datetime
from fastapi import FastAPI, File, UploadFile, HTTPException

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Welcome to Signature Detector API"}

class SignatureSpotDetector:
    def __init__(self):
        self.min_line_length = 40  # Reduced to catch shorter input fields
        self.min_date_length = 40
        self.empty_space_threshold = 240
        
        # Keywords for different types of inputs (added colon variants)
        self.input_types = {
            'signature': [
                'signature:', 'signature', 'sign:', 'sign', 'signed:', 'signed',
                'authorized:', 'approved:', 'by:', 'x:', 'sign here:'
            ],
            'name': [
                'name:', 'name', 'print:', 'printed:', 'full name:', 'first name:',
                'last name:', 'middle:', 'surname:'
            ],
            'date': [
                'date:', 'dated:', 'day:', 'month:', 'year:', 
                'effective:', 'effective date:', 'as of:'
            ],
            'phone': [
                'phone:', 'telephone:', 'tel:', 'cell:', 'mobile:', 'fax:',
                'contact:', 'number:', 'phone number:'
            ],
            'email': [
                'email:', 'e-mail:', '@', 'electronic:', 'contact:',
                'email address:', 'e-mail address:'
            ],
            'address': [
                'address:', 'street:', 'city:', 'state:', 'zip:', 'postal:',
                'country:', 'residence:'
            ],
            'title': [
                'title:', 'position:', 'job:', 'role:', 'designation:'
            ],
            'company': [
                'company:', 'organization:', 'business:', 'employer:',
                'corporation:', 'firm:'
            ]
        }
        self.print_keywords = [
            'print', 'printed', 'print name', 'type name',
            'printed name', 'name (print)', 'name (type)'
        ]
        # Add party-specific keywords
        self.contractor_keywords = [
            'corporation', 'inc', 'llc', 'seller', 'provider',
            'contractor', 'vendor', 'supplier', 'lessor', 'employer', 'investor'
        ]
        self.contractee_keywords = [
            'customer', 'buyer', 'purchaser', 'client',
            'lessee', 'employee', 'recipient', 'individual','company', 
        ]
        self.date_keywords = [
            'date', 'dated', 'day', 'month', 'year', 
            'mm/dd', 'dd/mm', 'today', 'fecha', 'data',
            'effective', 'effective date', 'as of',
            'executed', 'execution date', 'start date',
            'end date', 'termination date', 'on this',
            'commence', 'commencement'
        ]
        self.line_search_distance = 50
        self.current_date = datetime.now().strftime("%m/%d/%Y")
        self.signature_text = "TEST SIGNATURE"

    def identify_party(self, text, x, y, words, scale_x, scale_y):
        """Identify if signature is for contractor or contractee"""
        search_distance = 300
        
        # Get all text in the area, preserving order
        area_words = []
        for word in words:
            word_x = int(word[0] * scale_x)
            word_y = int(word[1] * scale_y)
            if (abs(word_y - y) <= search_distance and
                abs(word_x - x) <= search_distance):
                area_words.append({
                    'text': word[4],  # Keep original case
                    'x': word_x,
                    'y': word_y,
                    'distance': word_y - y  # Vertical distance only
                })
        
        # Sort by vertical position (top to bottom)
        area_words.sort(key=lambda w: w['y'])
        
        # Join text preserving case
        area_text = ' '.join(w['text'] for w in area_words)
        
        # Simple structural checks
        if 'COMPANY:' in area_text:
            # Check what comes immediately after "COMPANY:"
            company_parts = area_text.split('COMPANY:')
            if len(company_parts) > 1:
                company_name = company_parts[1].strip().split()[0:3]  # Get first few words after COMPANY:
                company_name = ' '.join(company_name)
                
                # If this is the company being invested in
                if any(word in company_name.upper() for word in ['SEMICONDUCTOR', 'TECH', 'SYSTEMS']):
                    return 'contractor'
                # If this is the investment company
                elif any(word in company_name.upper() for word in ['INVESTMENT', 'CAPITAL', 'VENTURES']):
                    return 'contractee'
        
        # Check for specific section headers
        if 'INVESTOR:' in area_text or 'PURCHASER:' in area_text:
            return 'contractee'
        if 'SELLER:' in area_text or 'PROVIDER:' in area_text:
            return 'contractor'
        
        # Look at the immediate context (closer words weighted more heavily)
        close_words = [w for w in area_words if abs(w['distance']) < 100]
        close_text = ' '.join(w['text'] for w in close_words)
        
        if 'CEO' in close_text or 'Title:' in close_text:
            # Check if we can find the company name nearby
            if 'SEMICONDUCTOR' in area_text or 'TECHNOLOGIES' in area_text:
                return 'contractor'
            if 'INVESTMENT' in area_text or 'VENTURES' in area_text:
                return 'contractee'
        
        return 'unknown'

    def check_keywords_with_priority(self, words, x, y, w, scale_x, scale_y, keywords):
        """Check for keywords with priority given to text on the left"""
        search_distance = self.line_search_distance
        
        # Collect words with their positions and calculate distances
        nearby_words = []
        for word in words:
            word_x = int(word[0] * scale_x)
            word_y = int(word[1] * scale_y)
            
            # Only consider words within vertical range
            if abs(word_y - y) <= search_distance:
                # Calculate horizontal distance (negative for left side)
                x_distance = word_x - x
                
                # Calculate priority score (lower is better)
                # Left side gets priority (multiply by 0.5)
                # Right side and vertical distance are secondary
                priority = abs(word_y - y)  # Vertical distance
                if x_distance < 0:  # Left side
                    priority += abs(x_distance) * 0.5
                else:  # Right side
                    priority += x_distance * 2
                
                nearby_words.append({
                    'text': word[4].lower(),
                    'priority': priority,
                    'is_left': x_distance < 0
                })
        
        # Sort by priority
        nearby_words.sort(key=lambda w: w['priority'])
        
        # Check words in priority order, giving preference to left side
        for word in nearby_words:
            if any(keyword in word['text'] for keyword in keywords):
                return True, word['is_left'], nearby_words
        
        return False, False, nearby_words

    def find_lines(self, page):
        signature_lines = []
        field_number = 1
        
        # Get page dimensions
        width = page.rect.width
        height = page.rect.height
        
        # Convert page to image for processing
        pix = page.get_pixmap(matrix=fitz.Matrix(300/72, 300/72))
        img_array = np.frombuffer(pix.samples, dtype=np.uint8)
        image = img_array.reshape(pix.height, pix.width, pix.n)
        
        if pix.n == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
        elif pix.n == 1:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        horizontal_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel)
        
        words = page.get_text("words")
        scale_x = image.shape[1] / width
        scale_y = image.shape[0] / height
        
        contours, _ = cv2.findContours(horizontal_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            
            # Convert coordinates back to PDF space
            pdf_x = x / scale_x
            pdf_y = y / scale_y
            pdf_w = w / scale_x
            pdf_h = h / scale_y
            
            # Check for empty space above and below
            above_region = gray[max(0, y-20):y, x:x+w]
            below_region = gray[y+h:min(y+h+10, gray.shape[0]), x:x+w]
            
            is_valid_line = (
                above_region.size > 0 and 
                below_region.size > 0 and 
                np.mean(above_region) > self.empty_space_threshold and
                np.mean(below_region) > self.empty_space_threshold
            )
            
            if is_valid_line:
                # Determine input type
                input_type, context = self.check_input_type(words, x, y, w, h, scale_x, scale_y)
                
                signature_lines.append({
                    'x': round(pdf_x, 2),
                    'y': round(pdf_y, 2),
                    'width': round(pdf_w, 2),
                    'height': round(pdf_h, 2),
                    'type': input_type,
                    'field_number': field_number,
                    'context': context
                })
                
                field_number += 1
        
        return {
            'signature_lines': signature_lines,
            'page_dimensions': {
                'width': width,
                'height': height
            }
        }

    def check_input_type(self, words, x, y, w, h, scale_x, scale_y):
        """Determine the type of input field and its context"""
        # Check for keywords near the line
        for input_type, keywords in self.input_types.items():
            found, is_left, nearby_words = self.check_keywords_with_priority(
                words, x, y, w, scale_x, scale_y, keywords
            )
            if found:
                # Get party information
                party = self.identify_party(
                    ' '.join(word['text'] for word in nearby_words),
                    x, y, words, scale_x, scale_y
                )
                
                # For signature fields, check if it's a print field
                if input_type == 'signature':
                    for word in nearby_words:
                        if any(pk in word['text'].lower() for pk in self.print_keywords):
                            input_type = 'print'
                            break
                
                # For date fields, add current date as default value
                default_value = self.current_date if input_type == 'date' else None
                
                # For signature fields, add test signature as default value
                if input_type == 'signature':
                    default_value = self.signature_text
                
                return input_type, {
                    'party': party,
                    'is_left_aligned': is_left,
                    'default_value': default_value
                }
        
        # If no specific type is found, check length for default categorization
        if w >= self.min_line_length:
            if w <= self.min_date_length:
                return 'date', {
                    'party': 'unknown',
                    'is_left_aligned': True,
                    'default_value': self.current_date
                }
            return 'signature', {
                'party': 'unknown',
                'is_left_aligned': True,
                'default_value': self.signature_text
            }
        
        return 'unknown', {
            'party': 'unknown',
            'is_left_aligned': True,
            'default_value': None
        }

@app.post("/process-pdf")
async def process_pdf_endpoint(file: UploadFile = File(...)):
    try:
        # Create a temporary file to store the uploaded PDF
        temp_file_path = f"temp_{file.filename}"
        with open(temp_file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Process the PDF
        detector = SignatureSpotDetector()
        pdf_document = fitz.open(temp_file_path)
        pages_results = []

        for page_num in range(len(pdf_document)):
            try:
                page = pdf_document[page_num]
                spots = detector.find_lines(page)
                
                pages_results.append({
                    'page_number': page_num + 1,
                    'signature_spots': spots
                })
            except Exception as page_error:
                raise HTTPException(
                    status_code=500,
                    detail=f"Error processing page {page_num + 1}: {str(page_error)}"
                )

        # Clean up
        pdf_document.close()
        os.remove(temp_file_path)

        return {
            'total_pages': len(pdf_document),
            'pages': pages_results
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 