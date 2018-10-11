from html.parser import HTMLParser

class Tag:
	def __init__(self, name):
		self.name = name
		self.categories = []
		self.children = []
		self.is_open = True
	
	def __repr__(self):	
		return self.name + "(" + (str(self.categories) if self.categories else "" ) + str(self.children) + ")"

	def name(self, name):
		self.name = name

	def close(self):
		self.is_open = False

	def set_categories(self, categories):
		self.categories = categories

	def set_name(self, name):
		self.name = name

class LabelParser(HTMLParser):
	def __init__(self):
		super().__init__()
		self.raw_document = ''
		self.document = []
		self.verbosity = 0

	def parse(self, raw_document):
		self.raw_document = raw_document
		super().feed(raw_document)
		validate_ast(self.document)
		return self.document

	def handle_starttag(self, tag_name, attrs):
		if self.verbosity > 1:
			print(str(self.getpos()) +":Encountered a start tag - Current status| : ", tag_name)
		
		if len(attrs) > 0:
			raise ParseError(str(self.getpos()) + ": tag has spaces in name")
	
		self.handle_starttag_helper(self.document, tag_name)	

	
	def handle_starttag_helper(self, on_list, tag_name):
		if len(on_list) < 1:
			on_list.append(Tag(tag_name))
			return		
		last_expression = on_list[-1]
	
		if type(last_expression) is str:
			on_list.append(Tag(tag_name))
		elif type(last_expression) is Tag:
			if not last_expression.is_open:
				on_list.append(Tag(tag_name))
			else: #last_expression is open
				self.handle_starttag_helper(last_expression.children, tag_name)


	def handle_endtag(self, tag_name):
		if self.verbosity > 1:
			print(str(self.getpos()) + ": Encountered an end tag| :", tag_name)
		
		self.handle_endtag_helper(self.document, tag_name)

	
	def handle_endtag_helper(self, on_list, tag_name):
		if len(on_list) < 1:
			raise ParseError(str(self.getpos()) + ": close tag mismatch:", tag_name)
		last_expression = on_list[-1]

		if type(last_expression) is str:
			raise  ParseError(str(self.getpos()) + ": encountered close tag ", tag_name, " without a matching open tag")
		elif type(last_expression) is Tag:
			if not last_expression.is_open:
				raise  ParseError(str(self.getpos()) + ": encountered close tag ", tag_name, " without an open tag")
			else: # last_expression is open
				if not last_expression.name == tag_name:
					self.handle_endtag_helper(last_expression.children, tag_name)	
				else:
					last_expression.close()


	def handle_data(self, text):
		if self.verbosity > 1:
			print(str(self.getpos()) + ": Encountered some text | :", text)
	
		self.handle_data_helper(self.document, text)

	
	def handle_data_helper(self, on_list, text):
		if len(on_list) < 1:
			on_list.append(text)
			return
		last_expression = on_list[-1]
		
		if type(last_expression) is str:
			raise ParseError(str(self.getpos()) + ": encountered a string " + text + " directly after another string")
		elif type(last_expression) is Tag:
			if not last_expression.is_open:
				on_list.append(text)
			else:
				#last expression is open
				self.handle_data_helper(last_expression.children, text)


""" Semantic validation takes place directly after parsing. The following things are checked:
1. Are the tag names valid (either made up of the categories, or 'data')  | invalid-tag-name
2. Are any top level <data> tags?  | no-lonely-data
3. Are there any tags nested in other tags?  | no-nested-tags

and converts to a new abstract syntax tree format (only difference is that the Tag class has an array of categories associatied with it)
"""

personal_categories = ['name','id-number', 'location', 'online-id', 'dob', 'phone', 'physical', 'physiological', 'professional', 'genetic', 'mental', 'economic', 'cultural', 'social']
sensitive_categories = ['criminal', 'origin', 'health', 'religion', 'political', 'philosophical', 'unions', 'sex-life', 'sex-orientation', 'biometric']
all_categories = personal_categories + sensitive_categories


def validate_ast(ast):
	for element in ast:		
		if type(element) is Tag:
			if not are_valid_categories(element.name.split("_")) and not element.name == 'data':
				raise SemanticError(element.name, " is not a valid category or data label")
		
			element.set_categories(element.name.split("_"))
			element.set_name("Label")

			if element.name == 'data':
				raise SemanticError("top-level data tag found")

			validate_children(element.children)

def validate_children(children):
	for element in children:
		if type(element) is str:
			continue
		elif type(element) is Tag:
			if are_valid_categories(element.name.split("_")):
				raise SemanticError("Label ", element.name, " cannot be nested under another label")
			elif not element.name == 'data':
				raise SemanticError("Label ", element.name, " is not recognized and can not be nested under another label")	
			elif element.name == 'data':
				for child in element.children:
					if type(child) is Tag and child.name == 'data':
						#If a data tag has a data child
						raise SemanticError("data tags cannot contain other data tags. Did you forget to close </data>?")


def are_valid_categories(categories):
	for category in categories:
		if category not in all_categories:
			return False
	return True


""" Exceptions """
class SemanticError(Exception):
	pass
class ParseError(Exception):
	pass





