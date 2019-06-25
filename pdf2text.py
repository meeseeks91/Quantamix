import os
import sys, getopt
from io import StringIO

from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage

#Convert the pdf cv to text
def pdf_to_text(file_name, pages=None):
    #In case there are mutiple pages
    if not pages:
        pagenumber = set()
    else:
        pagenumber = set(pages)

    output = StringIO()
    manager = PDFResourceManager()
    converter = TextConverter(manager, output, laparams=LAParams())
    interpreter = PDFPageInterpreter(manager, converter)

    input_file = open(file_name, 'rb')
    for page in PDFPage.get_pages(input_file, pagenumber):
        interpreter.process_page(page)
    input_file.close()
    converter.close()
    text = output.getvalue()
    output.close
    return text


#the pdf resumes are to be stored in the Candidates directory.

resume_directory = "Candidates"
text_directory = "Candidates_txt"

os.mkdir(text_directory)

directory = os.fsencode(resume_directory)

#loops through the directory and converts each resume into a text file to be stored in the text directory.
#This is also to ensure that the main directory in untouched.
for file in os.listdir(directory):
     filename = os.fsdecode(file)
     if filename.endswith(".pdf"):
         filepath = os.path.join(resume_directory, filename)
         text = pdf_to_text(filepath)
         targetpath = os.path.join(text_directory, filename)
         text_file = open(targetpath, "w")
         text_file.write(text)
         text_file.close()
     else:
         continue
