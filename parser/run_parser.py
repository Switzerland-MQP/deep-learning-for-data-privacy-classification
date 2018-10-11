from parser.parser import LabelParser
from parser.to_lines import flag_lines
import sys

def run_parser(filename):
	document = open(filename, "r")
	text = document.read()
	document.close()

	parser = LabelParser()
	ast = parser.parse(text)

	lines = flag_lines(ast)
	return lines


if __name__ == '__main__':
        if len(sys.argv) < 2:
                print("run_parser: must be run with a filename as argument")	
                quit()
        filename = sys.argv[1]
        lines = run_parser(filename)
        for line in lines:
                print(line)


