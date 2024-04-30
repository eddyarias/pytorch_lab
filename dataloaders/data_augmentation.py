'''
Choose Data Augmentation library and level of augmentation
'''

def data_aug_selector(args):
    if args.da_library in ['albumentations', 'Albumentations']:
        from dataloader.DA_albumentations import build_augmentation_sequence
        transform = build_augmentation_sequence(args.da_level)

    else:
        from dataloader.DA_torchvision import build_augmentation_sequence
        transform = build_augmentation_sequence(args.da_level)

    return transform
