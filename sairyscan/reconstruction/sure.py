import torch


class SureMap:
    """Calculate the Sure weights map for IFED and ISFED

    """
    def __init__(self):
        self.patch_size = 5

    def __call__(self, image_ref, image_a, image_b):
        width = image_ref.shape[0]
        height = image_ref.shape[1]
        image_ref = image_ref.view(1, 1, image_ref.shape[0], image_ref.shape[1])
        image_a = image_a.view(1, 1, image_a.shape[0], image_a.shape[1])
        image_b = image_b.view(1, 1, image_b.shape[0], image_b.shape[1])

        # add p//2 padding
        # todo

        # extract patch
        image_ref_patch = torch.nn.functional.unfold(image_ref, self.patch_size)
        image_a_patch = torch.nn.functional.unfold(image_a, self.patch_size)
        image_b_patch = torch.nn.functional.unfold(image_b, self.patch_size)

        # sure
        num = torch.sum((image_ref_patch-image_a_patch)*image_b_patch, dim=1)
        den = torch.sum(image_b_patch*image_b_patch, dim=1)
        sure_map = -num/den
        return sure_map.view(width-self.patch_size+1, height-self.patch_size+1)
