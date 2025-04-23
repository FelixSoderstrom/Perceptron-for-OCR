from pytorch.pipeline import run_training

if __name__ == "__main__":
    print("You are about to train a model.")
    print("This might take a while...")
    print("You may close the program mid-training by pressing Ctrl+C.")
    print("If the dataset is not present, it will be downloaded for you.")
    input("Press Enter to continue...")
    run_training()
    print(
        "The best performing model has been saved to "
        "'/pytorch-mnist-ocr/[ID]/checkpoints/best-checkpoint-epoch-[EPOCH]-[ACCURACY].ckpt'."
    )
    print(
        "Move it to '/checkpoints/' then run 'streamlit run app.py' to test it out!"
    )
