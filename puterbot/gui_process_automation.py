from segment_anything import SamPredictor, sam_model_registry,SamAutomaticMaskGenerator
from transformers import GPTJForCausalLM, GPT2Tokenizer
from paddleocr import PaddleOCR

from models import InputEvent

# Initialize Segment Anything model
sam = sam_model_registry["<model_type>"](checkpoint="<path/to/checkpoint>")
sam_predictor = SamPredictor(sam)

#Initialize Segment Anything model to generate masks for an entire image:
mask_generator = SamAutomaticMaskGenerator(sam)


# Initialize GPT-J model
tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-j-6B")
model = GPTJForCausalLM.from_pretrained("EleutherAI/gpt-j-6B")

# Initialize PaddleOCR model
ocr = PaddleOCR()

def generate_prompt(prev_masks, new_masks, new_text,prev_text):
    # Get the change between new and previous masks
    mask_diff = []
    for new_mask, prev_mask in zip(new_masks, prev_masks):
        diff = new_mask['segmentation'] - prev_mask['segmentation']
        mask_diff.append(diff)
        
    # Get the change between new and previous texts
    text_diff = set(new_text.split()) - set(prev_text.split())

    # Concatenate the mask and text differences into a single string prompt
    prompt = "Mask differences: " + " ".join([str(diff) for diff in mask_diff]) + "\n"
    prompt += "Text differences: " + " ".join(text_diff)

    return prompt

def predict_properties(prompt):
    # Generate text using GPT-J model
    inputs = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    outputs = model.generate(inputs, max_length=128, do_sample=True)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    ##############I dont know
    return generated_text


def parse_event(predicted_event):
    event_parts = predicted_event.split(",")
    event_dict = {}
    for part in event_parts:
        if "=" in part:
            key, value = part.split("=")
            event_dict[key.strip()] = value.strip()

    input_event = InputEvent()
    input_event.name = event_dict.get("name", "")
    input_event.timestamp = int(event_dict.get("timestamp", 0))
    input_event.recording_timestamp = int(event_dict.get("recording_timestamp", 0))
    input_event.screenshot_timestamp = int(event_dict.get("screenshot_timestamp", 0))
    input_event.window_event_timestamp = int(event_dict.get("window_event_timestamp", 0))
    input_event.mouse_x = float(event_dict.get("mouse_x", 0))
    input_event.mouse_y = float(event_dict.get("mouse_y", 0))
    input_event.mouse_dx = float(event_dict.get("mouse_dx", 0))
    input_event.mouse_dy = float(event_dict.get("mouse_dy", 0))
    input_event.mouse_button_name = event_dict.get("mouse_button_name", "")
    input_event.mouse_pressed = bool(event_dict.get("mouse_pressed", False))
    input_event.key_name = event_dict.get("key_name", "")
    input_event.key_char = event_dict.get("key_char", "")
    input_event.key_vk = event_dict.get("key_vk", "")
    input_event.canonical_key_name = event_dict.get("canonical_key_name", "")
    input_event.canonical_key_char = event_dict.get("canonical_key_char", "")
    input_event.canonical_key_vk = event_dict.get("canonical_key_vk", "")
    input_event.parent_id = int(event_dict.get("parent_id", 0))

    return input_event


def generate_input_event(new_screenshot, recording):
    #Get the latest screenshot object in the recording. 
    #.screenshots property is a list of InputEvent objects from a previous recording
    prev_screenshot = recording.screenshots.screenshot
    
    #Segment the objects in the new and previous screenshots
    prev_masks = mask_generator.generate(prev_screenshot.image)
    new_masks = mask_generator.generate(new_screenshot.image)
    
    #Extract text information from the new and previous screenshots
    new_text = ocr.ocr(new_screenshot.image)
    prev_text = ocr.ocr(prev_screenshot.image)
    
    #Generate textual prompts based on the segmented objects and extracted text
    prompt = generate_prompt(prev_masks, new_masks, new_text,prev_text)
    
    #Use the GPT-J model to predict the next InputeEvent properties
    predicted_event = predict_properties(prompt)
    
    #Create a new InputEvent object based on the predicted properties
    new_event = parse_event(predicted_event)
    
    return new_event
