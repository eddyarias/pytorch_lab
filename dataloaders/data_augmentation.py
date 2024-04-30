def data_aug_selector(args):
    if args.da_library in ['albumentations', 'Albumentations']:
        from dataloader.DA_albumentations import build_augmentation_sequence
        transform = build_augmentation_sequence(args.da_level)

    elif args.da_library in ['torchvision', 'pytorch', 'torch']:
        from dataloader.DA_torchvision import build_augmentation_sequence
        transform = build_augmentation_sequence(args.da_level)

    else: # imgaug
        from dataloader.DA_imgaug import build_augmentation_sequence
        transform = build_augmentation_sequence(args.da_level)

    return transform
