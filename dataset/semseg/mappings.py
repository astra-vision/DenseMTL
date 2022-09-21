class Mappings:
    Cityscapes = dict(
        # Standard Cityscapes 19 class set
        cls19 = {7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5, 19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11,
                 25: 12, 26: 13, 27: 14, 28: 15, 31: 16, 32: 17, 33: 18},

        # Standard Cityscapes 16 class set
        # cls16 = cls19 \ { truck = 22, terrain = 27, train = 31 }
        cls16 = {7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5, 19: 6, 20: 7, 21: 8, 23: 9, 24: 10, 25: 11,
                 26: 12, 28: 13, 32: 14, 33: 15},

        # Standard Cityscapes 13 class set
        # cls13 = cls16 \ { wall = 12, fence = 13, pole = 17 }
        cls13 = {7: 0, 8: 1, 11: 2, 19: 3, 20: 4, 21: 5, 23: 6, 24: 7, 25: 8, 26: 9, 28: 10, 32: 11,
                 33: 12},

        # "higher-level" Cityscapes mapping (7 classes)
        cls07 = {7: 0, 8: 0, 11: 1, 13: 1, 17: 2, 19: 2, 20: 2, 21: 3, 23: 4, 24: 5, 26: 6, 33: 6},

        # Our VKitti2/Cityscapes mapping (8 classes)
        cls08 = {7: 0, 11: 1, 17: 2, 19: 3, 20: 4, 21: 5, 23: 6, 26: 7, 28: 7})

    VKitti2 = dict(
        # Standard "full" 14 class set
        cls14 = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7,
                 8: 8, 9: 9, 10: 10, 11: 11, 12: 12, 13: 13},

        # Our VKitti2/Cityscapes mapping (8 classes)
        cls08 = {5: 0, 4: 1, 9: 2, 8: 3, 7: 4, 2: 5, 3: 5, 1: 6, 11: 7, 12: 7, 13: 7})

    Synthia = dict(
        # Synthia Cityscapes-compatible subset mapping (16 classes)""
        cls16 = {3: 0, 4: 1, 2: 2, 21: 3, 5: 4, 7: 5, 15: 6, 9: 7, 6: 8, 1: 9, 10: 10, 17: 11,
                 8: 12, 19: 13, 12: 14, 11: 15})