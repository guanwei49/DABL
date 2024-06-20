class ParsedLabel():
    def __init__(self, label, split, tags, bos, actions):
        self.label = label
        self.split_label = split
        self.tags = tags
        self.bos = bos
        self.actions = actions
