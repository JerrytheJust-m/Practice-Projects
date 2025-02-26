import torch
from torch.fx.experimental.migrate_gradual_types.constraint_transformation import transform_index_select

from main import SimpleCardClassifier
from glob import glob
from main import transform
from PIL import Image
import numpy as np
from main import target_to_class


if __name__ == "__main__":
    model = SimpleCardClassifier(num_classes = 53)
    model.load_state_dict(torch.load("PlayCard.pth", weights_only=True))
    model.eval()
    test_images = glob('archive/test/*/*')
    test_labels = glob('archive/test/*')
    test_example = np.random.choice(test_images, 1)

    #get image tensor
    image = Image.open(test_example[0]).convert("RGB")
    image_tensor = transform(image).unsqueeze(0)
   # image_tensor = image_tensor.to("mps")
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)
    print(test_example)
    #print(probabilities)
    #print(test_labels)
    #print(test_images)
    print(target_to_class[np.argmax(probabilities).item()])





