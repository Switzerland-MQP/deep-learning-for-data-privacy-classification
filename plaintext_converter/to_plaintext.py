import sys
import textract


def to_plaintext(filename):
    text = textract.process(filename)

    # Check if it converted an image pdf into one byte per page
    if len(text) < 30 and all(c is 12 for c in text):
        return textract.process(filename, method="tesseract")
    else:
        return text


#  if __name__ == '__main__':
    #  print(to_plaintext(sys.argv[1]))
