from pytorch_version.pytorch_pipeline import run_training

if __name__ == "__main__":
    print("\033c")
    input(
        "You are about to train the PyTorch network!\n"
        "If the learning data is not present, it will be downloaded.\n"
        "The learning might take a while depending on your machine.\n"
        "You may opt out at any time by pressing Ctrl+C.\n"
        "Press enter to continue..."
    )
    print("Training the network...")
    run_training()
    print("Training complete!")
