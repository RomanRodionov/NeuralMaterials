import torchvision.transforms.functional as TF
from datasets import load_dataset

#https://huggingface.co/datasets/gvecchio/MatSynth

def get_materials_dataset(res=(1024, 1024), streaming=True, license="CC0"):
    def process_img(x):
        x = TF.resize(x, res)
        x = TF.to_tensor(x)
        return x

    def process_batch(examples):
        examples["basecolor"] = [process_img(x) for x in examples["basecolor"]]
        return examples

    ds = load_dataset("gvecchio/MatSynth", streaming=streaming)
    ds = ds.shuffle(buffer_size=100)

    if license:
        ds = ds.filter(lambda x: x["metadata"]["license"] == license)

    ds = ds.map(process_batch, batched=True, batch_size=8)
    ds = ds.with_format("torch")
    
    return ds
