class ImageToText:
    def __init__(self, url):
        self.url = url

    def texting(self):
        prompt = "image generating"
        item_list = ["mouse", "desk mat", "mechanical keyboard", "led lamp", "pot plant"]

        return prompt, item_list