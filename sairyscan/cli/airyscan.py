import os
import argparse

from sairyscan.api import SAiryscanAPI
from skimage.io import imsave


def add_args_to_parser(parser, api):
    for filter_name in api.filters.get_keys():
        params = api.filters.get_parameters(filter_name)
        for key, value in params.items():
            parser.add_argument(f"--{key}", help=value['help'], default=value['default'])


def main():
    parser = argparse.ArgumentParser(description='SAiryscan reconstruction', conflict_handler='resolve')

    parser.add_argument('-i', '--input', help='Input image file', default='.czi')
    parser.add_argument('-r', '--reg', help='Registration method', default='none')
    parser.add_argument('-m', '--method', help='Reconstruction method', default='ISM')
    parser.add_argument('-e', '--enhance', help='Post processing enhancing', default='none')
    parser.add_argument('-o', '--output', help='Output image file', default='.tif')

    api = SAiryscanAPI()
    add_args_to_parser(parser, api)
    args = parser.parse_args()

    args_dict = vars(args)
    print('args=', vars(args))

    # instantiate pipeline
    if args.reg != 'none':
        reg_filter = api.filter(args.reg, **args_dict)
    else:
        reg_filter = None
    if args.method != 'none':
        rec_filter = api.filter(args.method, **args_dict)
    else:
        rec_filter = None
    if args.enhance != 'none':
        enhance_filter = api.filter(args.enhance, **args_dict)
    else:
        enhance_filter = None
    pipeline = api.pipeline(rec_filter, reg_filter, enhance_filter)

    # run pipeline
    reader = api.reader(args.input)
    out_image = pipeline(reader.data())
    imsave(args.output, out_image.detach().numpy())
