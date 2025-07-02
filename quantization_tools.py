import numpy as np

def evaluate_quantized_model(interpreter, dataset):
    input_index = interpreter.get_input_details()[0]['index']
    output_index = interpreter.get_output_details()[0]['index']
    
    correct = 0
    total = 0
    
    for images, labels in dataset:
        # Prepare batch (may need preprocessing depending on your model)
        batch_size = images.shape[0]
        for i in range(batch_size):
            # Set input tensor
            interpreter.set_tensor(input_index, images[i:i+1])
            
            # Run inference
            interpreter.invoke()
            
            # Get predictions
            predictions = interpreter.get_tensor(output_index)
            predicted_class = np.argmax(predictions)
            
            # Count correct predictions
            if predicted_class == labels[i]:
                correct += 1
            total += 1

    print("Current ", correct)
    print("Total ", total)
    
    return correct / total