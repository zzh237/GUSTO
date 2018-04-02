class GustoJasonConig():
    def __init__(self, jason):
        self.jason = jason
    def xmltodict(self):
        pass

    def json_deserializer(json_text, class_name):
        # sig = inspect.signature(class_name.__init__)
        # json_dicts = json.loads(json_text)
        #
        # if type(json_dicts) is list:
        #     return [_initialize_class_from_json(json_dict, class_name, sig.parameters) for json_dict in json_dicts]
        # elif type(json_dicts) is dict:
        #     return _initialize_class_from_json(json_dicts, class_name, sig.parameters)

        return None  # should make a new error and throw that when applicable