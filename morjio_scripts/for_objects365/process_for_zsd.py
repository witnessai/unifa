import os
import json 
import ipdb 

anno_root = '/data1/niehui/objects365/annotations/'
anno_names = ['val.json', 'train.json', ]
save_root = '/data1/niehui/objects365/annotations/zero_shot_detection'
# '/home/nieh/morjio/projects/detection/any_shot_detection/mmfewshot/data/coco/annotations/instances_train2014_seen_65_15.json'
# '/home/nieh/morjio/projects/detection/any_shot_detection/mmfewshot/data/coco/annotations/instances_train2014.json'

coco_super_class = ['person', 'vehicle', 'outdoor', 'animal', 'accessory', 'sports', 'kitchen', 'food', 'furniture', 'electronic', 'appliance', 'indoor']
objects365_paper_super_class = ['human and related accessories', 'living room', 'clothes', 'kitchen', 'instrument', 'transportation', 'bathroom', 'electronics', 'food', 'office supplies', 'animal']
objects365_web_super_class = ['human & accessory', 'clothes', 'traffic', 'living room',  'kitchen', 'office supplies & tools', 'electronics', 'foods', 'animal', 'sport', 'bathroom']
objects365_web_sub_class = [
    ['Person', 'Bracelet', 'Book', 'Flag', 'Ring', 'Necklace', 'Watch', 'Umbrella', 'Awning', 'Tent', 'Balloon', 'Stuffed Toy', 'Lantern', 'Cigar/Cigarette', 'Trophy', 'Cosmetics', 'Key', 'Medal', 'Kite', 'Lighter', 'Cosmetics Brush/Eyeliner Pencil', 'Lipstick', 'Cosmetics Mirror', 'Comb'],
    ['Other Shoes', 'Sneakers', 'Hat', 'Glasses', 'Handbag/Satchel', 'Gloves', 'Helmet', 'Leather Shoes', 'Boots', 'Belt', 'Tie', 'Backpack', 'Slippers', 'Sandals', 'Basket', 'High Heels', 'Skating shoes', 'Bow Tie', 'Luggage', 'Mask', 'Briefcase', 'Wallet/Purse'], 
    ['Car', 'Street Lights', 'Boat', 'suV', 'Traffic Light', 'Bicycle', 'Barrel/bucket', 'Van', 'Bus', 'Sailboat', 'Traffic cone', 'Motorcycle', 'Truck', 'Pickup Truck', 'Machinery Vehicle', 'Crane', 'Traffic Sign', 'Scooter', 'Train', 'Stop Sign', 'Sports Car', 'Trolley', 'Stroller', 'Heavy Truck', 'Airplane', 'Tricycle', 'Fire Truck', 'Fire Hydrant', 'Ambulance', 'Wheelchair', 'Carriage', 'Hotair balloon', 'Ship', 'Speed Limit Sign', 'Rickshaw', 'Parking meter', 'Helicopter', 'Hoverboard', 'Crosswalk Sign', 'Formula 1'], 
    ['Chair', 'Lamp', 'Desk', 'Storage box', 'Cabinet/shelf', 'Picture/Frame', 'Bench', 'Flower', 'Potted Plant', 'Vase', 'Pillow', 'Stool', 'Couch', 'Candle', 'Napkin', 'Air Conditioner', 'Power outlet', 'Carpet', 'Dining Table', 'Mirror', 'Clock', 'Hanger', 'Coffee Table', 'Fan', 'Bed', 'Side Table', 'Radiator', 'Nightstand'], 
    ['Bottle', 'cup', 'Plate', 'Bowl/Basin', 'Wine Glass', 'Knife', 'Fork', 'Spoon', 'Pot', 'Jug', 'Chopsticks', 'Refrigerator', 'Tea pot', 'Oven', 'Gas stove', 'Cutting/chopping Board', 'Microwave', 'Tong', 'Kettle', 'Extractor', 'Flask', 'Coffee Machine', 'Blender', 'Dishwasher', 'Induction Cooker', 'Rice Cooker', 'Toaster'],
    ['Trash bin Can', 'Pen/Pencil', 'Blackboard/Whiteboard', 'Marker', 'Telephone', 'Ladder', 'Shovel', 'Fire Extinguisher', 'Scissors', 'Paint Brush', 'Scale', 'Tape', 'Folder', 'Brush', 'Pliers', 'Board Eraser', 'Hammer', 'Screwdriver', 'Tape Measure/Ruler', 'Globe', 'Stapler', 'Calculator', 'Notepaper', 'Electric Drill', 'Pencil Case', 'Chainsaw', 'Eraser', 'Binoculars'], 
    ['Microphone', 'Speaker', 'Moniter/TV', 'Cell Phone', 'Tripod', 'Laptop', 'Camera', 'Keyboard', 'Head Phone', 'Surveillance Camera', 'Computer Box', 'Remote', 'Extension Cord', 'earphone', 'Tablet', 'Projector', 'Printer', 'Converter', 'Megaphone', 'Recorder', 'CD', 'Router/modem'],
    ['Canned', 'Bread', 'Green Vegetables', 'Pumpkin', 'Dessert', 'Tomato', 'Apple', 'Orange/Tangerine', 'Lemon', 'Strawberry', 'Pepper', 'Cake', 'Pie', 'Banana', 'Egg', 'Carrot', 'Grape', 'Potato', 'Broccoli', 'Cookies', 'Cucumber', 'French Fries', 'Green beans', 'Hamburger', 'Watermelon', 'Rice', 'Pizza', 'Cabbage', 'Ice cream', 'Sausage', 'Onion', 'Sushi', 'Peach', 'Green Onion', 'Pineapple', 'Cheese', 'Pasta', 'Lettuce', 'Pear', 'Candy', 'Garlic', 'Corn', 'Nuts', 'Chips', 'Mango', 'Eggplant', 'Kiwi fruit', 'Hot dog', 'Hamimelon', 'Steak', 'Mushroom', 'Sandwich', 'Cherry', 'Coconut', 'Avocado', 'Pomegranate', 'Donut', 'Plum', 'Papaya', 'Asparagus', 'Radish', 'Dumpling', 'Meat balls', 'Red Cabbage', 'Durian', 'Baozi', 'Grapefruit', 'Egg tart', 'Spring Rolls', 'Noodles', 'Okra'],
    ['wild Bird', 'Horse', 'Dog', 'Mouse', 'Other Fish', 'Duck', 'Cow', 'Sheep', 'Pigeon', 'Cat', 'Chicken', 'Elephant', 'Swan', 'Deer', 'Penguin', 'Shrimp', 'Goose', 'Pig', 'Crab', 'Seal', 'Lion', 'Parrot', 'Jellyfish', 'Oyster', 'Rabbit', 'Scallop', 'Camel', 'Goldfish', 'Zebra', 'Giraffe', 'Butterfly', 'Yak', 'Donkey', 'Dolphin', 'Antelope', 'Monkey', 'Lobster', 'Bear'], 
    ['Drum', 'Cymbal', 'Guitar', 'Lifesaver', 'Hockey Stick', 'Paddle', 'Baseball Glove', 'Other Balls', 'Piano', 'Soccer', 'Gun', 'Baseball Bat', 'Skiboard', 'Baseball', 'Golf Club', 'Billiards', 'Basketball', 'Saxophone', 'Violin', 'Snowboard', 'Swing', 'Cello', 'Skateboard', 'Surfboard', 'American Football', 'Slide', 'Trumpet', 'Flute', 'Poker Card', 'Golf Ball', 'Trombone', 'Fishing Rod', 'Frisbee', 'Volleyball', 'Tennis Racket', 'Cue', 'Tennis', 'Hurdle', 'Tuba', 'Game board', 'Table Tennis paddle', 'French', 'Target', 'Treadmill', 'Dumbbell', 'Curling', 'Barbell', 'Table Tennis'], 
    ['Faucet', 'Sink', 'Towel', 'Toiletry', 'Toilet Paper', 'Cleaning Products', 'Tissue', 'Toilet', 'Bathtub', 'Toothbrush', 'Showerhead', 'Broom', 'Soap', 'Washing Machine/Drying Machine', 'Mop', 'Urinal', 'Hair Dryer']
]
objects365_web_sub_class = [x.lower() for item in objects365_web_sub_class for x in item]
objects365_json_categories = ['scale', 'tape', 'chicken', 'hurdle', 'game board', 'baozi', 'target', 'plants pot/vase', 'toothbrush', 'projector', 'cheese', 'candy', 'durian', 'dumbbell', 'gas stove', 'lion', 'french fries', 'bench', 'power outlet', 'faucet', 'storage box', 'crab', 'helicopter', 'chainsaw', 'antelope', 'hamimelon', 'jellyfish', 'kettle', 'marker', 'clutch', 'lettuce', 'toilet', 'oven', 'baseball', 'drum', 'hanger', 'toaster', 'bracelet', 'cherry', 'tissue', 'watermelon', 'basketball', 'cleaning products', 'tent', 'fire hydrant', 'truck', 'rice cooker', 'microscope', 'tablet', 'stuffed animal', 'golf ball', 'CD', 'eggplant', 'bowl', 'desk', 'eagle', 'slippers', 'horn', 'carpet', 'notepaper', 'peach', 'saw', 'surfboard', 'facial cleanser', 'corn', 'folder', 'violin', 'watch', 'glasses', 'shampoo/shower gel', 'pizza', 'asparagus', 'mushroom', 'steak', 'suitcase', 'table tennis paddle', 'mango', 'boots', 'necklace', 'noodles', 'volleyball', 'baseball bat', 'nuts', 'stroller', 'pumpkin', 'strawberry', 'pear', 'luggage', 'sandals', 'liquid soap', 'handbag', 'flashlight', 'trombone', 'remote', 'shovel', 'ladder', 'cake', 'pomegranate', 'clock', 'vent', 'cymbal', 'iron', 'okra', 'pasta', 'lantern', 'broom', 'fire extinguisher', 'snowboard', 'rice', 'swing', 'cow', 'van', 'tuba', 'book', 'swan', 'lamp', 'race car', 'egg', 'avocado', 'guitar', 'radio', 'sneakers', 'eraser', 'measuring cup', 'sushi', 'deer', 'parrot', 'scissors', 'balloon', 'tortoise/turtle', 'meat balls', 'cat', 'electric drill', 'comb', 'sausage', 'bar soap', 'hamburger', 'pepper', 'router/modem', 'spring rolls', 'american football', 'egg tart', 'tape measure/ruler', 'banana', 'gun', 'billiards', 'picture/frame', 'paper towel', 'bus', 'goldfish', 'computer box', 'potted plant', 'ship', 'ambulance', 'dog', 'medal', 'butterfly', 'hair dryer', 'globe', 'french horn', 'board eraser', 'tea pot', 'telephone', 'mop', 'broccoli', 'dolphin', 'chair', 'hat', 'tripod', 'traffic light', 'hot dog', 'pot/pan', 'car', 'dining table', 'crosswalk sign', 'tomato', 'barrel/bucket', 'washing machine', 'polar bear', 'tie', 'monkey', 'green beans', 'cucumber', 'cookies', 'suv', 'brush', 'carrot', 'tennis racket', 'helmet', 'sink', 'stool', 'flower', 'radiator', 'fishing rod', 'Life saver', 'lighter', 'bread', 'radish', 'human', 'traffic cone', 'knife', 'grapes', 'cellphone', 'trophy', 'urinal', 'cup', 'paint brush', 'mouse', 'soccer', 'cutting/chopping board', 'wheelchair', 'Accordion/keyboard/piano', 'goose', 'red cabbage', 'plate', 'saxophone', 'laptop', 'facial mask', 'onion', 'motorbike/motorcycle', 'canned', 'lobster', 'toiletries', 'earphone', 'flag', 'Bread/bun', 'trumpet', 'parking meter', 'garlic', 'skateboard', 'pie', 'barbell', 'yak', 'stapler', 'tangerine', 'zebra', 'traffic sign', 'bottle', 'hotair balloon', 'sailboat', 'llama', 'blackboard/whiteboard', 'coffee machine', 'flute', 'pencil case', 'ice cream', 'combine with bowl', 'kite', 'microphone', 'fork', 'hoverboard', 'blender', 'skating and skiing shoes', 'nightstand', 'toothpaste', 'poker card', 'fan', 'orange', 'chopsticks', 'pig', 'bathtub', 'glove', 'golf club', 'refrigerator', 'rickshaw', 'candle', 'mirror', 'microwave', 'converter', 'airplane', 'lemon', 'head phone', 'tricycle', 'bear', 'backpack', 'apple', 'trolley', 'tong', 'papaya', 'cello', 'camel', 'binoculars', 'cabbage', 'umbrella', 'cigar', 'pomelo', 'cabinet/shelf', 'keyboard', 'horse', 'duck', 'combine with glove', 'pineapple', 'potato', 'air conditioner', 'pliers', 'fire truck', 'hockey stick', 'elephant', 'sports car', 'toy', 'mangosteen', 'rabbit', 'bicycle', 'giraffe', 'screwdriver', 'spoon', 'sheep', 'key', 'wine glass', 'treadmill', 'extension cord', 'shrimp', 'ring', 'boat', 'green vegetables', 'coffee table', 'pitaya', 'shark', 'basket', 'wild bird', 'carriage', 'slide', 'fish', 'frisbee', 'hammer', 'printer', 'plum', 'towel/napkin', 'camera', 'speaker', 'pickup truck', 'high heels', 'bow tie', 'pigeon', 'coconut', 'machinery vehicle', 'sofa', 'bed', 'tennis ball', 'dates', 'street lights', 'paddle', 'calculator', 'starfish', 'chips', 'train', 'kiwi fruit', 'belt', 'monitor', 'skis', 'leather shoes', 'sandwich', 'Electronic stove and gas stove', 'penguin', 'surveillance camera', 'cue', 'scallop', 'green onion', 'seal', 'crane', 'donkey', 'pen/pencil', 'donut', 'pillow', 'trash bin/can']
objects365_json_categories = [x.lower() for x in objects365_json_categories]

web2json_for_diff_classname = {'toiletry':'toiletries', 'cigar/cigarette':'cigar', 'cell phone': 'cellphone', 'toilet paper':'paper towel',  'motorcycle':'motorbike/motorcycle', 'trash bin can':'trash bin/can', 'orange/tangerine':'orange', 'person':'human', 'grape':'grapes', 'vase':'plants pot/vase', 'mask':'facial mask', 'skating shoes':'skating and skiing shoes', 'piano':'accordion/keyboard/piano', 'soap':'bar soap', 'washing machine/drying machine':'washing machine', 'meat ball':'meat balls', 'lifesaver':'life saver', 'pot':'pot/pan', 'tennis':'tennis ball', 'moniter/tv':'monitor','skiboard':'skis', 'recorder':'radio', 'formula 1':'race car', 'grapefruit':'pomelo', 'handbag/satchel':'handbag', 'wallet/purse':'clutch', 'bowl/basin':'bowl', 'couch':'sofa', 'induction cooker':'electronic stove and gas stove', 'gloves':'glove', 'megaphone':'horn'}
same_clsname_for_web = []
same_clsname_for_json = []
for classname in objects365_web_sub_class:
    if classname in web2json_for_diff_classname.keys():
        oldname = classname
        classname = web2json_for_diff_classname[classname]
        if classname in objects365_json_categories:
            same_clsname_for_web.append(oldname)
            same_clsname_for_json.append(classname)
    else:
        if classname in objects365_json_categories:
            same_clsname_for_web.append(classname)
            same_clsname_for_json.append(classname)


objects365_json_categories = list(set(objects365_json_categories) - set(same_clsname_for_json))
objects365_web_sub_class = list(set(objects365_web_sub_class) - set(same_clsname_for_web))
print(objects365_json_categories)
print(objects365_web_sub_class)
print(len(objects365_json_categories), len(objects365_web_sub_class))
# web2json_for_different_name = {
#     toiletry:toiletries
# }
ipdb.set_trace()
for name in anno_names:
    anno_path = os.path.join(anno_root, name)
    with open(anno_path) as fd:
        data = json.load(fd)
    cats = data['categories']
    cat_names = []
    for cat in cats:
        if cat['name'] == 'tissue ':
            print(cat)
            ipdb.set_trace()
    new_data = dict()
    new_data['licenses'] = None
    new_data['info'] = None
    new_data['type'] = None

    # print(data.keys())
    # dict_keys(['images', 'annotations', 'categories', 'licenses', 'info', 'type'])
    
    ipdb.set_trace()
    
    
    break


## shell commandd:
## CUDA_VISIBLE_DEVICES=4 python process_for_zsd.py