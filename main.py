# main.py
import argparse
from data_module import load_data, split_data
from train_module import initialize_network, train_model
from evaluate_module import load_model, evaluate_model

def main():
    parser = argparse.ArgumentParser(description="Segmentation Pipeline CLI")
    subparsers = parser.add_subparsers(dest='command', help='sub-command help')

    parser_train = subparsers.add_parser('train', help='train the model')
    parser_train.add_argument('--data-dir', type=str, required=True, help='directory with training data')

    parser_evaluate = subparsers.add_parser('evaluate', help='evaluate the model')
    parser_evaluate.add_argument('--model-path', type=str, required=True, help='path to trained model file')

    args = parser.parse_args()

    if args.command == 'train':
        files = load_data(args.data_dir)
        train_files, test_files = split_data(files)
        # Implement the training logic: Create data loaders and then train the model

    elif args.command == 'evaluate':
        model = load_model(args.model_path)
        # Implement evaluation logic: Load test data, create data loaders, and evaluate the model

if __name__ == "__main__":
    main()
