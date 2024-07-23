import yaml
from ultralytics import YOLO


# def create_yaml1(yaml_path):
#     with open(yaml_path, 'w') as f:
#         yaml.dump(dataset, f, default_flow_style=False)

def create_yaml(dataset, yaml_path):

    with open(yaml_path, 'w') as f:
        yaml.dump(dataset, f, default_flow_style=False)

if __name__ == '__main__':

    train_image_dir = '/datashare/HW1/labeled_image_data/images/train'
    train_label_dir = '/datashare/HW1/labeled_image_data/labels/train'
    val_image_dir = '/datashare/HW1/labeled_image_data/images/val'
    val_label_dir = '/datashare/HW1/labeled_image_data/labels/val'
    pseudo_dir = './pseudo_data/images'

    dataset = {
        'train': train_image_dir,
        'val': val_image_dir,
        'nc': 3,
        'names': ['Empty', 'Tweezers', 'Needle_driver']
    }

    combined_dataset = {
            'train': pseudo_dir,
            'val': val_image_dir,
            'nc': 3,
            'names': ['Empty', 'Tweezers', 'Needle_driver']
        }
    
    create_yaml(dataset, './dataset.yaml')
    create_yaml(combined_dataset, './combined_dataset.yaml')