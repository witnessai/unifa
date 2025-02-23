import numpy as np

IMAGENET_ALL_CLASSES = [
    'accordion', 'airplane', 'ant', 'antelope', 'apple', 'armadillo',
    'artichoke', 'axe', 'baby bed', 'backpack', 'bagel', 'balance beam',
    'banana', 'band aid', 'banjo', 'baseball', 'basketball', 'bathing cap',
    'beaker', 'bear', 'bee', 'bell pepper', 'bench', 'bicycle', 'binder',
    'bird', 'bookshelf', 'bow tie', 'bow', 'bowl', 'brassiere', 'burrito',
    'bus', 'butterfly', 'camel', 'can opener', 'car', 'cart', 'cattle',
    'cello', 'centipede', 'chain saw', 'chair', 'chime', 'cocktail shaker',
    'coffee maker', 'computer keyboard', 'computer mouse', 'corkscrew',
    'cream', 'croquet ball', 'crutch', 'cucumber', 'cup or mug', 'diaper',
    'digital clock', 'dishwasher', 'dog', 'domestic cat', 'dragonfly',
    'drum', 'dumbbell', 'electric fan', 'elephant', 'face powder', 'fig',
    'filing cabinet', 'flower pot', 'flute', 'fox', 'french horn', 'frog',
    'frying pan', 'giant panda', 'goldfish', 'golf ball', 'golfcart',
    'guacamole', 'guitar', 'hair dryer', 'hair spray', 'hamburger',
    'hammer', 'hamster', 'harmonica', 'harp', 'hat with a wide brim',
    'head cabbage', 'helmet', 'hippopotamus', 'horizontal bar', 'horse',
    'hotdog', 'iPod', 'isopod', 'jellyfish', 'koala bear', 'ladle',
    'ladybug', 'lamp', 'laptop', 'lemon', 'lion', 'lipstick', 'lizard',
    'lobster', 'maillot', 'maraca', 'microphone', 'microwave', 'milk can',
    'miniskirt', 'monkey', 'motorcycle', 'mushroom', 'nail', 'neck brace',
    'oboe', 'orange', 'otter', 'pencil box', 'pencil sharpener', 'perfume',
    'person', 'piano', 'pineapple', 'ping-pong ball', 'pitcher', 'pizza',
    'plastic bag', 'plate rack', 'pomegranate', 'popsicle', 'porcupine',
    'power drill', 'pretzel', 'printer', 'puck', 'punching bag', 'purse',
    'rabbit', 'racket', 'ray', 'red panda', 'refrigerator',
    'remote control', 'rubber eraser', 'rugby ball', 'ruler',
    'salt or pepper shaker', 'saxophone', 'scorpion', 'screwdriver',
    'seal', 'sheep', 'ski', 'skunk', 'snail', 'snake', 'snowmobile',
    'snowplow', 'soap dispenser', 'soccer ball', 'sofa', 'spatula',
    'squirrel', 'starfish', 'stethoscope', 'stove', 'strainer',
    'strawberry', 'stretcher', 'sunglasses', 'swimming trunks', 'swine',
    'syringe', 'table', 'tape player', 'tennis ball', 'tick', 'tie',
    'tiger', 'toaster', 'traffic light', 'train', 'trombone', 'trumpet',
    'turtle', 'tv or monitor', 'unicycle', 'vacuum', 'violin',
    'volleyball', 'waffle iron', 'washer', 'water bottle', 'watercraft',
    'whale', 'wine bottle', 'zebra'
]

IMAGENET_UNSEEN_CLASSES = ['bench',
    'bow tie',
    'burrito',
    'can opener',
    'dishwasher',
    'electric fan',
    'golf ball',
    'hamster',
    'harmonica',
    'horizontal bar',
    'iPod',
    'maraca',
    'pencil box',
    'pineapple',
    'plate rack',
    'ray',
    'scorpion',
    'snail',
    'swimming trunks',
    'syringe',
    'tiger',
    'train',
    'unicycle'
]

VOC_ALL_CLASSES = np.array([
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus','cat',
    'chair', 'cow', 'diningtable', 'horse', 'motorbike','person', 
    'pottedplant', 'sheep', 'tvmonitor','car', 'dog', 'sofa', 'train'
])

## for voc few-shot, the order is base + novel
VOC_ALL_CLASSES_SPLIT1 = np.array([
    'aeroplane', 'bicycle', 'boat', 'bottle', 'car', 'cat', 'chair', 'diningtable', 'dog', 'horse', 'person', 'pottedplant', 'sheep', 'train', 'tvmonitor', 'bird', 'bus', 'cow', 'motorbike', 'sofa'])

VOC_ALL_CLASSES_SPLIT2 = np.array([
    'bicycle', 'bird', 'boat', 'bus', 'car', 'cat', 'chair', 'diningtable', 'dog', 'motorbike', 'person', 'pottedplant', 'sheep', 'train', 'tvmonitor', 'aeroplane', 'bottle', 'cow', 'horse', 'sofa'
])

VOC_ALL_CLASSES_SPLIT3 = np.array([
    'aeroplane', 'bicycle', 'bird', 'bottle', 'bus', 'car', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'person', 'pottedplant', 'train', 'tvmonitor', 'boat', 'cat', 'motorbike', 'sheep', 'sofa'
])

VOC_BASE_CLASSES_SPLIT1 = np.array([
    'aeroplane', 'bicycle', 'boat', 'bottle', 'car', 'cat', 'chair', 'diningtable', 'dog', 'horse', 'person', 'pottedplant', 'sheep', 'train', 'tvmonitor'
])

VOC_BASE_CLASSES_SPLIT2 = np.array([
    'bicycle', 'bird', 'boat', 'bus', 'car', 'cat', 'chair', 'diningtable', 'dog', 'motorbike', 'person', 'pottedplant', 'sheep', 'train', 'tvmonitor'
])

VOC_BASE_CLASSES_SPLIT3 = np.array([
    'aeroplane', 'bicycle', 'bird', 'bottle', 'bus', 'car', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'person', 'pottedplant', 'train', 'tvmonitor'
])

VOC_NOVEL_CLASSES_SPLIT1 = np.array([
    'bird', 'bus', 'cow', 'motorbike', 'sofa'
])
VOC_NOVEL_CLASSES_SPLIT2 = np.array([
    'aeroplane', 'bottle', 'cow', 'horse', 'sofa'
])
VOC_NOVEL_CLASSES_SPLIT3 = np.array([
    'boat', 'cat', 'motorbike', 'sheep', 'sofa'
])

COCO_ALL_CLASSES = np.array([
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
    'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
    'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
    'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
    'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
    'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
    'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
    'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
])

COCO_UNSEEN_CLASSES_65_15 = np.array([
    'airplane', 'train', 'parking meter', 'cat', 'bear', 
    'suitcase', 'frisbee', 'snowboard', 'fork', 'sandwich', 
    'hot dog', 'toilet', 'mouse', 'toaster', 'hair drier'
])

LVIS_ALL_CLASSES = np.array([
    'aerosol_can', 'air_conditioner', 'airplane', 'alarm_clock', 'alcohol', 'alligator', 'almond', 'ambulance', 'amplifier', 'anklet', 'antenna', 'apple', 'applesauce', 'apricot', 'apron', 'aquarium', 'arctic_(type_of_shoe)', 'armband', 'armchair', 'armoire', 'armor', 'artichoke', 'trash_can', 'ashtray', 'asparagus', 'atomizer', 'avocado', 'award', 'awning', 'ax', 'baboon', 'baby_buggy', 'basketball_backboard', 'backpack', 'handbag', 'suitcase', 'bagel', 'bagpipe', 'baguet', 'bait', 'ball', 'ballet_skirt', 'balloon', 'bamboo', 'banana', 'Band_Aid', 'bandage', 'bandanna', 'banjo', 'banner', 'barbell', 'barge', 'barrel', 'barrette', 'barrow', 'baseball_base', 'baseball', 'baseball_bat', 'baseball_cap', 'baseball_glove', 'basket', 'basketball', 'bass_horn', 'bat_(animal)', 'bath_mat', 'bath_towel', 'bathrobe', 'bathtub', 'batter_(food)', 'battery', 'beachball', 'bead', 'bean_curd', 'beanbag', 'beanie', 'bear', 'bed', 'bedpan', 'bedspread', 'cow', 'beef_(food)', 'beeper', 'beer_bottle', 'beer_can', 'beetle', 'bell', 'bell_pepper', 'belt', 'belt_buckle', 'bench', 'beret', 'bib', 'Bible', 'bicycle', 'visor', 'billboard', 'binder', 'binoculars', 'bird', 'birdfeeder', 'birdbath', 'birdcage', 'birdhouse', 'birthday_cake', 'birthday_card', 'pirate_flag', 'black_sheep', 'blackberry', 'blackboard', 'blanket', 'blazer', 'blender', 'blimp', 'blinker', 'blouse', 'blueberry', 'gameboard', 'boat', 'bob', 'bobbin', 'bobby_pin', 'boiled_egg', 'bolo_tie', 'deadbolt', 'bolt', 'bonnet', 'book', 'bookcase', 'booklet', 'bookmark', 'boom_microphone', 'boot', 'bottle', 'bottle_opener', 'bouquet', 'bow_(weapon)', 'bow_(decorative_ribbons)', 'bow-tie', 'bowl', 'pipe_bowl', 'bowler_hat', 'bowling_ball', 'box', 'boxing_glove', 'suspenders', 'bracelet', 'brass_plaque', 'brassiere', 'bread-bin', 'bread', 'breechcloth', 'bridal_gown', 'briefcase', 'broccoli', 'broach', 'broom', 'brownie', 'brussels_sprouts', 'bubble_gum', 'bucket', 'horse_buggy', 'bull', 'bulldog', 'bulldozer', 'bullet_train', 'bulletin_board', 'bulletproof_vest', 'bullhorn', 'bun', 'bunk_bed', 'buoy', 'burrito', 'bus_(vehicle)', 'business_card', 'butter', 'butterfly', 'button', 'cab_(taxi)', 'cabana', 'cabin_car', 'cabinet', 'locker', 'cake', 'calculator', 'calendar', 'calf', 'camcorder', 'camel', 'camera', 'camera_lens', 'camper_(vehicle)', 'can', 'can_opener', 'candle', 'candle_holder', 'candy_bar', 'candy_cane', 'walking_cane', 'canister', 'canoe', 'cantaloup', 'canteen', 'cap_(headwear)', 'bottle_cap', 'cape', 'cappuccino', 'car_(automobile)', 'railcar_(part_of_a_train)', 'elevator_car', 'car_battery', 'identity_card', 'card', 'cardigan', 'cargo_ship', 'carnation', 'horse_carriage', 'carrot', 'tote_bag', 'cart', 'carton', 'cash_register', 'casserole', 'cassette', 'cast', 'cat', 'cauliflower', 'cayenne_(spice)', 'CD_player', 'celery', 'cellular_telephone', 'chain_mail', 'chair', 'chaise_longue', 'chalice', 'chandelier', 'chap', 'checkbook', 'checkerboard', 'cherry', 'chessboard', 'chicken_(animal)', 'chickpea', 'chili_(vegetable)', 'chime', 'chinaware', 'crisp_(potato_chip)', 'poker_chip', 'chocolate_bar', 'chocolate_cake', 'chocolate_milk', 'chocolate_mousse', 'choker', 'chopping_board', 'chopstick', 'Christmas_tree', 'slide', 'cider', 'cigar_box', 'cigarette', 'cigarette_case', 'cistern', 'clarinet', 'clasp', 'cleansing_agent', 'cleat_(for_securing_rope)', 'clementine', 'clip', 'clipboard', 'clippers_(for_plants)', 'cloak', 'clock', 'clock_tower', 'clothes_hamper', 'clothespin', 'clutch_bag', 'coaster', 'coat', 'coat_hanger', 'coatrack', 'cock', 'cockroach', 'cocoa_(beverage)', 'coconut', 'coffee_maker', 'coffee_table', 'coffeepot', 'coil', 'coin', 'colander', 'coleslaw', 'coloring_material', 'combination_lock', 'pacifier', 'comic_book', 'compass', 'computer_keyboard', 'condiment', 'cone', 'control', 'convertible_(automobile)', 'sofa_bed', 'cooker', 'cookie', 'cooking_utensil', 'cooler_(for_food)', 'cork_(bottle_plug)', 'corkboard', 'corkscrew', 'edible_corn', 'cornbread', 'cornet', 'cornice', 'cornmeal', 'corset', 'costume', 'cougar', 'coverall', 'cowbell', 'cowboy_hat', 'crab_(animal)', 'crabmeat', 'cracker', 'crape', 'crate', 'crayon', 'cream_pitcher', 'crescent_roll', 'crib', 'crock_pot', 'crossbar', 'crouton', 'crow', 'crowbar', 'crown', 'crucifix', 'cruise_ship', 'police_cruiser', 'crumb', 'crutch', 'cub_(animal)', 'cube', 'cucumber', 'cufflink', 'cup', 'trophy_cup', 'cupboard', 'cupcake', 'hair_curler', 'curling_iron', 'curtain', 'cushion', 'cylinder', 'cymbal', 'dagger', 'dalmatian', 'dartboard', 'date_(fruit)', 'deck_chair', 'deer', 'dental_floss', 'desk', 'detergent', 'diaper', 'diary', 'die', 'dinghy', 'dining_table', 'tux', 'dish', 'dish_antenna', 'dishrag', 'dishtowel', 'dishwasher', 'dishwasher_detergent', 'dispenser', 'diving_board', 'Dixie_cup', 'dog', 'dog_collar', 'doll', 'dollar', 'dollhouse', 'dolphin', 'domestic_ass', 'doorknob', 'doormat', 'doughnut', 'dove', 'dragonfly', 'drawer', 'underdrawers', 'dress', 'dress_hat', 'dress_suit', 'dresser', 'drill', 'drone', 'dropper', 'drum_(musical_instrument)', 'drumstick', 'duck', 'duckling', 'duct_tape', 'duffel_bag', 'dumbbell', 'dumpster', 'dustpan', 'eagle', 'earphone', 'earplug', 'earring', 'easel', 'eclair', 'eel', 'egg', 'egg_roll', 'egg_yolk', 'eggbeater', 'eggplant', 'electric_chair', 'refrigerator', 'elephant', 'elk', 'envelope', 'eraser', 'escargot', 'eyepatch', 'falcon', 'fan', 'faucet', 'fedora', 'ferret', 'Ferris_wheel', 'ferry', 'fig_(fruit)', 'fighter_jet', 'figurine', 'file_cabinet', 'file_(tool)', 'fire_alarm', 'fire_engine', 'fire_extinguisher', 'fire_hose', 'fireplace', 'fireplug', 'first-aid_kit', 'fish', 'fish_(food)', 'fishbowl', 'fishing_rod', 'flag', 'flagpole', 'flamingo', 'flannel', 'flap', 'flash', 'flashlight', 'fleece', 'flip-flop_(sandal)', 'flipper_(footwear)', 'flower_arrangement', 'flute_glass', 'foal', 'folding_chair', 'food_processor', 'football_(American)', 'football_helmet', 'footstool', 'fork', 'forklift', 'freight_car', 'French_toast', 'freshener', 'frisbee', 'frog', 'fruit_juice', 'frying_pan', 'fudge', 'funnel', 'futon', 'gag', 'garbage', 'garbage_truck', 'garden_hose', 'gargle', 'gargoyle', 'garlic', 'gasmask', 'gazelle', 'gelatin', 'gemstone', 'generator', 'giant_panda', 'gift_wrap', 'ginger', 'giraffe', 'cincture', 'glass_(drink_container)', 'globe', 'glove', 'goat', 'goggles', 'goldfish', 'golf_club', 'golfcart', 'gondola_(boat)', 'goose', 'gorilla', 'gourd', 'grape', 'grater', 'gravestone', 'gravy_boat', 'green_bean', 'green_onion', 'griddle', 'grill', 'grits', 'grizzly', 'grocery_bag', 'guitar', 'gull', 'gun', 'hairbrush', 'hairnet', 'hairpin', 'halter_top', 'ham', 'hamburger', 'hammer', 'hammock', 'hamper', 'hamster', 'hair_dryer', 'hand_glass', 'hand_towel', 'handcart', 'handcuff', 'handkerchief', 'handle', 'handsaw', 'hardback_book', 'harmonium', 'hat', 'hatbox', 'veil', 'headband', 'headboard', 'headlight', 'headscarf', 'headset', 'headstall_(for_horses)', 'heart', 'heater', 'helicopter', 'helmet', 'heron', 'highchair', 'hinge', 'hippopotamus', 'hockey_stick', 'hog', 'home_plate_(baseball)', 'honey', 'fume_hood', 'hook', 'hookah', 'hornet', 'horse', 'hose', 'hot-air_balloon', 'hotplate', 'hot_sauce', 'hourglass', 'houseboat', 'hummingbird', 'hummus', 'polar_bear', 'icecream', 'popsicle', 'ice_maker', 'ice_pack', 'ice_skate', 'igniter', 'inhaler', 'iPod', 'iron_(for_clothing)', 'ironing_board', 'jacket', 'jam', 'jar', 'jean', 'jeep', 'jelly_bean', 'jersey', 'jet_plane', 'jewel', 'jewelry', 'joystick', 'jumpsuit', 'kayak', 'keg', 'kennel', 'kettle', 'key', 'keycard', 'kilt', 'kimono', 'kitchen_sink', 'kitchen_table', 'kite', 'kitten', 'kiwi_fruit', 'knee_pad', 'knife', 'knitting_needle', 'knob', 'knocker_(on_a_door)', 'koala', 'lab_coat', 'ladder', 'ladle', 'ladybug', 'lamb_(animal)', 'lamb-chop', 'lamp', 'lamppost', 'lampshade', 'lantern', 'lanyard', 'laptop_computer', 'lasagna', 'latch', 'lawn_mower', 'leather', 'legging_(clothing)', 'Lego', 'legume', 'lemon', 'lemonade', 'lettuce', 'license_plate', 'life_buoy', 'life_jacket', 'lightbulb', 'lightning_rod', 'lime', 'limousine', 'lion', 'lip_balm', 'liquor', 'lizard', 'log', 'lollipop', 'speaker_(stero_equipment)', 'loveseat', 'machine_gun', 'magazine', 'magnet', 'mail_slot', 'mailbox_(at_home)', 'mallard', 'mallet', 'mammoth', 'manatee', 'mandarin_orange', 'manger', 'manhole', 'map', 'marker', 'martini', 'mascot', 'mashed_potato', 'masher', 'mask', 'mast', 'mat_(gym_equipment)', 'matchbox', 'mattress', 'measuring_cup', 'measuring_stick', 'meatball', 'medicine', 'melon', 'microphone', 'microscope', 'microwave_oven', 'milestone', 'milk', 'milk_can', 'milkshake', 'minivan', 'mint_candy', 'mirror', 'mitten', 'mixer_(kitchen_tool)', 'money', 'monitor_(computer_equipment) computer_monitor', 'monkey', 'motor', 'motor_scooter', 'motor_vehicle', 'motorcycle', 'mound_(baseball)', 'mouse_(computer_equipment)', 'mousepad', 'muffin', 'mug', 'mushroom', 'music_stool', 'musical_instrument', 'nailfile', 'napkin', 'neckerchief', 'necklace', 'necktie', 'needle', 'nest', 'newspaper', 'newsstand', 'nightshirt', 'nosebag_(for_animals)', 'noseband_(for_animals)', 'notebook', 'notepad', 'nut', 'nutcracker', 'oar', 'octopus_(food)', 'octopus_(animal)', 'oil_lamp', 'olive_oil', 'omelet', 'onion', 'orange_(fruit)', 'orange_juice', 'ostrich', 'ottoman', 'oven', 'overalls_(clothing)', 'owl', 'packet', 'inkpad', 'pad', 'paddle', 'padlock', 'paintbrush', 'painting', 'pajamas', 'palette', 'pan_(for_cooking)', 'pan_(metal_container)', 'pancake', 'pantyhose', 'papaya', 'paper_plate', 'paper_towel', 'paperback_book', 'paperweight', 'parachute', 'parakeet', 'parasail_(sports)', 'parasol', 'parchment', 'parka', 'parking_meter', 'parrot', 'passenger_car_(part_of_a_train)', 'passenger_ship', 'passport', 'pastry', 'patty_(food)', 'pea_(food)', 'peach', 'peanut_butter', 'pear', 'peeler_(tool_for_fruit_and_vegetables)', 'wooden_leg', 'pegboard', 'pelican', 'pen', 'pencil', 'pencil_box', 'pencil_sharpener', 'pendulum', 'penguin', 'pennant', 'penny_(coin)', 'pepper', 'pepper_mill', 'perfume', 'persimmon', 'person', 'pet', 'pew_(church_bench)', 'phonebook', 'phonograph_record', 'piano', 'pickle', 'pickup_truck', 'pie', 'pigeon', 'piggy_bank', 'pillow', 'pin_(non_jewelry)', 'pineapple', 'pinecone', 'ping-pong_ball', 'pinwheel', 'tobacco_pipe', 'pipe', 'pistol', 'pita_(bread)', 'pitcher_(vessel_for_liquid)', 'pitchfork', 'pizza', 'place_mat', 'plate', 'platter', 'playpen', 'pliers', 'plow_(farm_equipment)', 'plume', 'pocket_watch', 'pocketknife', 'poker_(fire_stirring_tool)', 'pole', 'polo_shirt', 'poncho', 'pony', 'pool_table', 'pop_(soda)', 'postbox_(public)', 'postcard', 'poster', 'pot', 'flowerpot', 'potato', 'potholder', 'pottery', 'pouch', 'power_shovel', 'prawn', 'pretzel', 'printer', 'projectile_(weapon)', 'projector', 'propeller', 'prune', 'pudding', 'puffer_(fish)', 'puffin', 'pug-dog', 'pumpkin', 'puncher', 'puppet', 'puppy', 'quesadilla', 'quiche', 'quilt', 'rabbit', 'race_car', 'racket', 'radar', 'radiator', 'radio_receiver', 'radish', 'raft', 'rag_doll', 'raincoat', 'ram_(animal)', 'raspberry', 'rat', 'razorblade', 'reamer_(juicer)', 'rearview_mirror', 'receipt', 'recliner', 'record_player', 'reflector', 'remote_control', 'rhinoceros', 'rib_(food)', 'rifle', 'ring', 'river_boat', 'road_map', 'robe', 'rocking_chair', 'rodent', 'roller_skate', 'Rollerblade', 'rolling_pin', 'root_beer', 'router_(computer_equipment)', 'rubber_band', 'runner_(carpet)', 'plastic_bag', 'saddle_(on_an_animal)', 'saddle_blanket', 'saddlebag', 'safety_pin', 'sail', 'salad', 'salad_plate', 'salami', 'salmon_(fish)', 'salmon_(food)', 'salsa', 'saltshaker', 'sandal_(type_of_shoe)', 'sandwich', 'satchel', 'saucepan', 'saucer', 'sausage', 'sawhorse', 'saxophone', 'scale_(measuring_instrument)', 'scarecrow', 'scarf', 'school_bus', 'scissors', 'scoreboard', 'scraper', 'screwdriver', 'scrubbing_brush', 'sculpture', 'seabird', 'seahorse', 'seaplane', 'seashell', 'sewing_machine', 'shaker', 'shampoo', 'shark', 'sharpener', 'Sharpie', 'shaver_(electric)', 'shaving_cream', 'shawl', 'shears', 'sheep', 'shepherd_dog', 'sherbert', 'shield', 'shirt', 'shoe', 'shopping_bag', 'shopping_cart', 'short_pants', 'shot_glass', 'shoulder_bag', 'shovel', 'shower_head', 'shower_cap', 'shower_curtain', 'shredder_(for_paper)', 'signboard', 'silo', 'sink', 'skateboard', 'skewer', 'ski', 'ski_boot', 'ski_parka', 'ski_pole', 'skirt', 'skullcap', 'sled', 'sleeping_bag', 'sling_(bandage)', 'slipper_(footwear)', 'smoothie', 'snake', 'snowboard', 'snowman', 'snowmobile', 'soap', 'soccer_ball', 'sock', 'sofa', 'softball', 'solar_array', 'sombrero', 'soup', 'soup_bowl', 'soupspoon', 'sour_cream', 'soya_milk', 'space_shuttle', 'sparkler_(fireworks)', 'spatula', 'spear', 'spectacles', 'spice_rack', 'spider', 'crawfish', 'sponge', 'spoon', 'sportswear', 'spotlight', 'squid_(food)', 'squirrel', 'stagecoach', 'stapler_(stapling_machine)', 'starfish', 'statue_(sculpture)', 'steak_(food)', 'steak_knife', 'steering_wheel', 'stepladder', 'step_stool', 'stereo_(sound_system)', 'stew', 'stirrer', 'stirrup', 'stool', 'stop_sign', 'brake_light', 'stove', 'strainer', 'strap', 'straw_(for_drinking)', 'strawberry', 'street_sign', 'streetlight', 'string_cheese', 'stylus', 'subwoofer', 'sugar_bowl', 'sugarcane_(plant)', 'suit_(clothing)', 'sunflower', 'sunglasses', 'sunhat', 'surfboard', 'sushi', 'mop', 'sweat_pants', 'sweatband', 'sweater', 'sweatshirt', 'sweet_potato', 'swimsuit', 'sword', 'syringe', 'Tabasco_sauce', 'table-tennis_table', 'table', 'table_lamp', 'tablecloth', 'tachometer', 'taco', 'tag', 'taillight', 'tambourine', 'army_tank', 'tank_(storage_vessel)', 'tank_top_(clothing)', 'tape_(sticky_cloth_or_paper)', 'tape_measure', 'tapestry', 'tarp', 'tartan', 'tassel', 'tea_bag', 'teacup', 'teakettle', 'teapot', 'teddy_bear', 'telephone', 'telephone_booth', 'telephone_pole', 'telephoto_lens', 'television_camera', 'television_set', 'tennis_ball', 'tennis_racket', 'tequila', 'thermometer', 'thermos_bottle', 'thermostat', 'thimble', 'thread', 'thumbtack', 'tiara', 'tiger', 'tights_(clothing)', 'timer', 'tinfoil', 'tinsel', 'tissue_paper', 'toast_(food)', 'toaster', 'toaster_oven', 'toilet', 'toilet_tissue', 'tomato', 'tongs', 'toolbox', 'toothbrush', 'toothpaste', 'toothpick', 'cover', 'tortilla', 'tow_truck', 'towel', 'towel_rack', 'toy', 'tractor_(farm_equipment)', 'traffic_light', 'dirt_bike', 'trailer_truck', 'train_(railroad_vehicle)', 'trampoline', 'tray', 'trench_coat', 'triangle_(musical_instrument)', 'tricycle', 'tripod', 'trousers', 'truck', 'truffle_(chocolate)', 'trunk', 'vat', 'turban', 'turkey_(food)', 'turnip', 'turtle', 'turtleneck_(clothing)', 'typewriter', 'umbrella', 'underwear', 'unicycle', 'urinal', 'urn', 'vacuum_cleaner', 'vase', 'vending_machine', 'vent', 'vest', 'videotape', 'vinegar', 'violin', 'vodka', 'volleyball', 'vulture', 'waffle', 'waffle_iron', 'wagon', 'wagon_wheel', 'walking_stick', 'wall_clock', 'wall_socket', 'wallet', 'walrus', 'wardrobe', 'washbasin', 'automatic_washer', 'watch', 'water_bottle', 'water_cooler', 'water_faucet', 'water_heater', 'water_jug', 'water_gun', 'water_scooter', 'water_ski', 'water_tower', 'watering_can', 'watermelon', 'weathervane', 'webcam', 'wedding_cake', 'wedding_ring', 'wet_suit', 'wheel', 'wheelchair', 'whipped_cream', 'whistle', 'wig', 'wind_chime', 'windmill', 'window_box_(for_plants)', 'windshield_wiper', 'windsock', 'wine_bottle', 'wine_bucket', 'wineglass', 'blinder_(for_horses)', 'wok', 'wolf', 'wooden_spoon', 'wreath', 'wrench', 'wristband', 'wristlet', 'yacht', 'yogurt', 'yoke_(animal_equipment)', 'zebra', 'zucchini'
])

LVIS_BASE_CLASSES = np.array([
    'aerosol_can', 'air_conditioner', 'airplane', 'alarm_clock', 'alcohol', 'alligator', 'almond', 'ambulance', 'amplifier', 'anklet', 'antenna', 'apple', 'apron', 'aquarium', 'armband', 'armchair', 'artichoke', 'trash_can', 'ashtray', 'asparagus', 'atomizer', 'avocado', 'award', 'awning', 'baby_buggy', 'basketball_backboard', 'backpack', 'handbag', 'suitcase', 'bagel', 'ball', 'balloon', 'bamboo', 'banana', 'Band_Aid', 'bandage', 'bandanna', 'banner', 'barrel', 'barrette', 'barrow', 'baseball_base', 'baseball', 'baseball_bat', 'baseball_cap', 'baseball_glove', 'basket', 'basketball', 'bat_(animal)', 'bath_mat', 'bath_towel', 'bathrobe', 'bathtub', 'battery', 'bead', 'bean_curd', 'beanbag', 'beanie', 'bear', 'bed', 'bedspread', 'cow', 'beef_(food)', 'beer_bottle', 'beer_can', 'bell', 'bell_pepper', 'belt', 'belt_buckle', 'bench', 'beret', 'bib', 'bicycle', 'visor', 'billboard', 'binder', 'binoculars', 'bird', 'birdfeeder', 'birdbath', 'birdcage', 'birdhouse', 'birthday_cake', 'black_sheep', 'blackberry', 'blackboard', 'blanket', 'blazer', 'blender', 'blinker', 'blouse', 'blueberry', 'boat', 'bobbin', 'bobby_pin', 'boiled_egg', 'deadbolt', 'bolt', 'book', 'bookcase', 'booklet', 'boot', 'bottle', 'bottle_opener', 'bouquet', 'bow_(decorative_ribbons)', 'bow-tie', 'bowl', 'bowler_hat', 'box', 'suspenders', 'bracelet', 'brassiere', 'bread-bin', 'bread', 'bridal_gown', 'briefcase', 'broccoli', 'broom', 'brownie', 'brussels_sprouts', 'bucket', 'bull', 'bulldog', 'bullet_train', 'bulletin_board', 'bullhorn', 'bun', 'bunk_bed', 'buoy', 'bus_(vehicle)', 'business_card', 'butter', 'butterfly', 'button', 'cab_(taxi)', 'cabin_car', 'cabinet', 'cake', 'calculator', 'calendar', 'calf', 'camcorder', 'camel', 'camera', 'camera_lens', 'camper_(vehicle)', 'can', 'can_opener', 'candle', 'candle_holder', 'candy_cane', 'walking_cane', 'canister', 'canoe', 'cantaloup', 'cap_(headwear)', 'bottle_cap', 'cape', 'cappuccino', 'car_(automobile)', 'railcar_(part_of_a_train)', 'identity_card', 'card', 'cardigan', 'horse_carriage', 'carrot', 'tote_bag', 'cart', 'carton', 'cash_register', 'cast', 'cat', 'cauliflower', 'cayenne_(spice)', 'CD_player', 'celery', 'cellular_telephone', 'chair', 'chandelier', 'cherry', 'chicken_(animal)', 'chickpea', 'chili_(vegetable)', 'crisp_(potato_chip)', 'chocolate_bar', 'chocolate_cake', 'choker', 'chopping_board', 'chopstick', 'Christmas_tree', 'slide', 'cigarette', 'cigarette_case', 'cistern', 'clasp', 'cleansing_agent', 'clip', 'clipboard', 'clock', 'clock_tower', 'clothes_hamper', 'clothespin', 'coaster', 'coat', 'coat_hanger', 'coatrack', 'cock', 'coconut', 'coffee_maker', 'coffee_table', 'coffeepot', 'coin', 'colander', 'coleslaw', 'pacifier', 'computer_keyboard', 'condiment', 'cone', 'control', 'cookie', 'cooler_(for_food)', 'cork_(bottle_plug)', 'corkscrew', 'edible_corn', 'cornet', 'cornice', 'corset', 'costume', 'cowbell', 'cowboy_hat', 'crab_(animal)', 'cracker', 'crate', 'crayon', 'crescent_roll', 'crib', 'crock_pot', 'crossbar', 'crow', 'crown', 'crucifix', 'cruise_ship', 'police_cruiser', 'crumb', 'crutch', 'cub_(animal)', 'cube', 'cucumber', 'cufflink', 'cup', 'trophy_cup', 'cupboard', 'cupcake', 'curtain', 'cushion', 'dartboard', 'deck_chair', 'deer', 'dental_floss', 'desk', 'diaper', 'dining_table', 'dish', 'dish_antenna', 'dishrag', 'dishtowel', 'dishwasher', 'dispenser', 'Dixie_cup', 'dog', 'dog_collar', 'doll', 'dolphin', 'domestic_ass', 'doorknob', 'doormat', 'doughnut', 'drawer', 'underdrawers', 'dress', 'dress_hat', 'dress_suit', 'dresser', 'drill', 'drum_(musical_instrument)', 'duck', 'duckling', 'duct_tape', 'duffel_bag', 'dumpster', 'eagle', 'earphone', 'earring', 'easel', 'egg', 'egg_yolk', 'eggbeater', 'eggplant', 'refrigerator', 'elephant', 'elk', 'envelope', 'eraser', 'fan', 'faucet', 'Ferris_wheel', 'ferry', 'fighter_jet', 'figurine', 'file_cabinet', 'fire_alarm', 'fire_engine', 'fire_extinguisher', 'fire_hose', 'fireplace', 'fireplug', 'fish', 'fish_(food)', 'fishing_rod', 'flag', 'flagpole', 'flamingo', 'flannel', 'flap', 'flashlight', 'flip-flop_(sandal)', 'flipper_(footwear)', 'flower_arrangement', 'flute_glass', 'foal', 'folding_chair', 'food_processor', 'football_(American)', 'footstool', 'fork', 'forklift', 'freight_car', 'French_toast', 'freshener', 'frisbee', 'frog', 'fruit_juice', 'frying_pan', 'garbage_truck', 'garden_hose', 'gargle', 'garlic', 'gazelle', 'gelatin', 'giant_panda', 'gift_wrap', 'ginger', 'giraffe', 'cincture', 'glass_(drink_container)', 'globe', 'glove', 'goat', 'goggles', 'golf_club', 'golfcart', 'goose', 'grape', 'grater', 'gravestone', 'green_bean', 'green_onion', 'grill', 'grizzly', 'grocery_bag', 'guitar', 'gull', 'gun', 'hairbrush', 'hairnet', 'hairpin', 'ham', 'hamburger', 'hammer', 'hammock', 'hamster', 'hair_dryer', 'hand_towel', 'handcart', 'handkerchief', 'handle', 'hat', 'veil', 'headband', 'headboard', 'headlight', 'headscarf', 'headstall_(for_horses)', 'heart', 'heater', 'helicopter', 'helmet', 'highchair', 'hinge', 'hog', 'home_plate_(baseball)', 'honey', 'fume_hood', 'hook', 'horse', 'hose', 'hot_sauce', 'hummingbird', 'polar_bear', 'icecream', 'ice_maker', 'igniter', 'iPod', 'iron_(for_clothing)', 'ironing_board', 'jacket', 'jam', 'jar', 'jean', 'jeep', 'jersey', 'jet_plane', 'jewelry', 'jumpsuit', 'kayak', 'kettle', 'key', 'kilt', 'kimono', 'kitchen_sink', 'kite', 'kitten', 'kiwi_fruit', 'knee_pad', 'knife', 'knob', 'ladder', 'ladle', 'ladybug', 'lamb_(animal)', 'lamp', 'lamppost', 'lampshade', 'lantern', 'lanyard', 'laptop_computer', 'latch', 'legging_(clothing)', 'Lego', 'lemon', 'lettuce', 'license_plate', 'life_buoy', 'life_jacket', 'lightbulb', 'lime', 'lion', 'lip_balm', 'lizard', 'log', 'lollipop', 'speaker_(stero_equipment)', 'loveseat', 'magazine', 'magnet', 'mail_slot', 'mailbox_(at_home)', 'mandarin_orange', 'manger', 'manhole', 'map', 'marker', 'mashed_potato', 'mask', 'mast', 'mat_(gym_equipment)', 'mattress', 'measuring_cup', 'measuring_stick', 'meatball', 'medicine', 'melon', 'microphone', 'microwave_oven', 'milk', 'minivan', 'mirror', 'mitten', 'mixer_(kitchen_tool)', 'money', 'monitor_(computer_equipment) computer_monitor', 'monkey', 'motor', 'motor_scooter', 'motorcycle', 'mound_(baseball)', 'mouse_(computer_equipment)', 'mousepad', 'muffin', 'mug', 'mushroom', 'musical_instrument', 'napkin', 'necklace', 'necktie', 'needle', 'nest', 'newspaper', 'newsstand', 'nightshirt', 'noseband_(for_animals)', 'notebook', 'notepad', 'nut', 'oar', 'oil_lamp', 'olive_oil', 'onion', 'orange_(fruit)', 'orange_juice', 'ostrich', 'ottoman', 'oven', 'overalls_(clothing)', 'owl', 'packet', 'pad', 'paddle', 'padlock', 'paintbrush', 'painting', 'pajamas', 'palette', 'pan_(for_cooking)', 'pancake', 'paper_plate', 'paper_towel', 'parachute', 'parakeet', 'parasail_(sports)', 'parasol', 'parka', 'parking_meter', 'parrot', 'passenger_car_(part_of_a_train)', 'passport', 'pastry', 'pea_(food)', 'peach', 'peanut_butter', 'pear', 'peeler_(tool_for_fruit_and_vegetables)', 'pelican', 'pen', 'pencil', 'penguin', 'pepper', 'pepper_mill', 'perfume', 'person', 'pet', 'pew_(church_bench)', 'phonograph_record', 'piano', 'pickle', 'pickup_truck', 'pie', 'pigeon', 'pillow', 'pineapple', 'pinecone', 'pipe', 'pita_(bread)', 'pitcher_(vessel_for_liquid)', 'pizza', 'place_mat', 'plate', 'platter', 'pliers', 'pocketknife', 'poker_(fire_stirring_tool)', 'pole', 'polo_shirt', 'pony', 'pop_(soda)', 'postbox_(public)', 'postcard', 'poster', 'pot', 'flowerpot', 'potato', 'potholder', 'pottery', 'pouch', 'power_shovel', 'prawn', 'pretzel', 'printer', 'projectile_(weapon)', 'projector', 'propeller', 'pumpkin', 'puppy', 'quilt', 'rabbit', 'racket', 'radiator', 'radio_receiver', 'radish', 'raft', 'raincoat', 'ram_(animal)', 'raspberry', 'razorblade', 'reamer_(juicer)', 'rearview_mirror', 'receipt', 'recliner', 'record_player', 'reflector', 'remote_control', 'rhinoceros', 'rifle', 'ring', 'robe', 'rocking_chair', 'rolling_pin', 'router_(computer_equipment)', 'rubber_band', 'runner_(carpet)', 'plastic_bag', 'saddle_(on_an_animal)', 'saddle_blanket', 'saddlebag', 'sail', 'salad', 'salami', 'salmon_(fish)', 'salsa', 'saltshaker', 'sandal_(type_of_shoe)', 'sandwich', 'saucer', 'sausage', 'scale_(measuring_instrument)', 'scarf', 'school_bus', 'scissors', 'scoreboard', 'screwdriver', 'scrubbing_brush', 'sculpture', 'seabird', 'seahorse', 'seashell', 'sewing_machine', 'shaker', 'shampoo', 'shark', 'shaving_cream', 'sheep', 'shield', 'shirt', 'shoe', 'shopping_bag', 'shopping_cart', 'short_pants', 'shoulder_bag', 'shovel', 'shower_head', 'shower_curtain', 'signboard', 'silo', 'sink', 'skateboard', 'skewer', 'ski', 'ski_boot', 'ski_parka', 'ski_pole', 'skirt', 'sled', 'sleeping_bag', 'slipper_(footwear)', 'snowboard', 'snowman', 'snowmobile', 'soap', 'soccer_ball', 'sock', 'sofa', 'solar_array', 'soup', 'soupspoon', 'sour_cream', 'spatula', 'spectacles', 'spice_rack', 'spider', 'sponge', 'spoon', 'sportswear', 'spotlight', 'squirrel', 'stapler_(stapling_machine)', 'starfish', 'statue_(sculpture)', 'steak_(food)', 'steering_wheel', 'step_stool', 'stereo_(sound_system)', 'stirrup', 'stool', 'stop_sign', 'brake_light', 'stove', 'strainer', 'strap', 'straw_(for_drinking)', 'strawberry', 'street_sign', 'streetlight', 'suit_(clothing)', 'sunflower', 'sunglasses', 'sunhat', 'surfboard', 'sushi', 'mop', 'sweat_pants', 'sweatband', 'sweater', 'sweatshirt', 'sweet_potato', 'swimsuit', 'sword', 'table', 'table_lamp', 'tablecloth', 'tag', 'taillight', 'tank_(storage_vessel)', 'tank_top_(clothing)', 'tape_(sticky_cloth_or_paper)', 'tape_measure', 'tapestry', 'tarp', 'tartan', 'tassel', 'tea_bag', 'teacup', 'teakettle', 'teapot', 'teddy_bear', 'telephone', 'telephone_booth', 'telephone_pole', 'television_camera', 'television_set', 'tennis_ball', 'tennis_racket', 'thermometer', 'thermos_bottle', 'thermostat', 'thread', 'thumbtack', 'tiara', 'tiger', 'tights_(clothing)', 'timer', 'tinfoil', 'tinsel', 'tissue_paper', 'toast_(food)', 'toaster', 'toaster_oven', 'toilet', 'toilet_tissue', 'tomato', 'tongs', 'toolbox', 'toothbrush', 'toothpaste', 'toothpick', 'cover', 'tortilla', 'tow_truck', 'towel', 'towel_rack', 'toy', 'tractor_(farm_equipment)', 'traffic_light', 'dirt_bike', 'trailer_truck', 'train_(railroad_vehicle)', 'tray', 'tricycle', 'tripod', 'trousers', 'truck', 'trunk', 'turban', 'turkey_(food)', 'turtle', 'turtleneck_(clothing)', 'typewriter', 'umbrella', 'underwear', 'urinal', 'urn', 'vacuum_cleaner', 'vase', 'vending_machine', 'vent', 'vest', 'videotape', 'volleyball', 'waffle', 'wagon', 'wagon_wheel', 'walking_stick', 'wall_clock', 'wall_socket', 'wallet', 'automatic_washer', 'watch', 'water_bottle', 'water_cooler', 'water_faucet', 'water_jug', 'water_scooter', 'water_ski', 'water_tower', 'watering_can', 'watermelon', 'weathervane', 'webcam', 'wedding_cake', 'wedding_ring', 'wet_suit', 'wheel', 'wheelchair', 'whipped_cream', 'whistle', 'wig', 'wind_chime', 'windmill', 'window_box_(for_plants)', 'windshield_wiper', 'windsock', 'wine_bottle', 'wine_bucket', 'wineglass', 'blinder_(for_horses)', 'wok', 'wooden_spoon', 'wreath', 'wrench', 'wristband', 'wristlet', 'yacht', 'yogurt', 'yoke_(animal_equipment)', 'zebra', 'zucchini'
])


LVIS_NOVEL_CLASSES = np.array([
    'applesauce', 'apricot', 'arctic_(type_of_shoe)', 'armoire', 'armor', 'ax', 'baboon', 'bagpipe', 'baguet', 'bait', 'ballet_skirt', 'banjo', 'barbell', 'barge', 'bass_horn', 'batter_(food)', 'beachball', 'bedpan', 'beeper', 'beetle', 'Bible', 'birthday_card', 'pirate_flag', 'blimp', 'gameboard', 'bob', 'bolo_tie', 'bonnet', 'bookmark', 'boom_microphone', 'bow_(weapon)', 'pipe_bowl', 'bowling_ball', 'boxing_glove', 'brass_plaque', 'breechcloth', 'broach', 'bubble_gum', 'horse_buggy', 'bulldozer', 'bulletproof_vest', 'burrito', 'cabana', 'locker', 'candy_bar', 'canteen', 'elevator_car', 'car_battery', 'cargo_ship', 'carnation', 'casserole', 'cassette', 'chain_mail', 'chaise_longue', 'chalice', 'chap', 'checkbook', 'checkerboard', 'chessboard', 'chime', 'chinaware', 'poker_chip', 'chocolate_milk', 'chocolate_mousse', 'cider', 'cigar_box', 'clarinet', 'cleat_(for_securing_rope)', 'clementine', 'clippers_(for_plants)', 'cloak', 'clutch_bag', 'cockroach', 'cocoa_(beverage)', 'coil', 'coloring_material', 'combination_lock', 'comic_book', 'compass', 'convertible_(automobile)', 'sofa_bed', 'cooker', 'cooking_utensil', 'corkboard', 'cornbread', 'cornmeal', 'cougar', 'coverall', 'crabmeat', 'crape', 'cream_pitcher', 'crouton', 'crowbar', 'hair_curler', 'curling_iron', 'cylinder', 'cymbal', 'dagger', 'dalmatian', 'date_(fruit)', 'detergent', 'diary', 'die', 'dinghy', 'tux', 'dishwasher_detergent', 'diving_board', 'dollar', 'dollhouse', 'dove', 'dragonfly', 'drone', 'dropper', 'drumstick', 'dumbbell', 'dustpan', 'earplug', 'eclair', 'eel', 'egg_roll', 'electric_chair', 'escargot', 'eyepatch', 'falcon', 'fedora', 'ferret', 'fig_(fruit)', 'file_(tool)', 'first-aid_kit', 'fishbowl', 'flash', 'fleece', 'football_helmet', 'fudge', 'funnel', 'futon', 'gag', 'garbage', 'gargoyle', 'gasmask', 'gemstone', 'generator', 'goldfish', 'gondola_(boat)', 'gorilla', 'gourd', 'gravy_boat', 'griddle', 'grits', 'halter_top', 'hamper', 'hand_glass', 'handcuff', 'handsaw', 'hardback_book', 'harmonium', 'hatbox', 'headset', 'heron', 'hippopotamus', 'hockey_stick', 'hookah', 'hornet', 'hot-air_balloon', 'hotplate', 'hourglass', 'houseboat', 'hummus', 'popsicle', 'ice_pack', 'ice_skate', 'inhaler', 'jelly_bean', 'jewel', 'joystick', 'keg', 'kennel', 'keycard', 'kitchen_table', 'knitting_needle', 'knocker_(on_a_door)', 'koala', 'lab_coat', 'lamb-chop', 'lasagna', 'lawn_mower', 'leather', 'legume', 'lemonade', 'lightning_rod', 'limousine', 'liquor', 'machine_gun', 'mallard', 'mallet', 'mammoth', 'manatee', 'martini', 'mascot', 'masher', 'matchbox', 'microscope', 'milestone', 'milk_can', 'milkshake', 'mint_candy', 'motor_vehicle', 'music_stool', 'nailfile', 'neckerchief', 'nosebag_(for_animals)', 'nutcracker', 'octopus_(food)', 'octopus_(animal)', 'omelet', 'inkpad', 'pan_(metal_container)', 'pantyhose', 'papaya', 'paperback_book', 'paperweight', 'parchment', 'passenger_ship', 'patty_(food)', 'wooden_leg', 'pegboard', 'pencil_box', 'pencil_sharpener', 'pendulum', 'pennant', 'penny_(coin)', 'persimmon', 'phonebook', 'piggy_bank', 'pin_(non_jewelry)', 'ping-pong_ball', 'pinwheel', 'tobacco_pipe', 'pistol', 'pitchfork', 'playpen', 'plow_(farm_equipment)', 'plume', 'pocket_watch', 'poncho', 'pool_table', 'prune', 'pudding', 'puffer_(fish)', 'puffin', 'pug-dog', 'puncher', 'puppet', 'quesadilla', 'quiche', 'race_car', 'radar', 'rag_doll', 'rat', 'rib_(food)', 'river_boat', 'road_map', 'rodent', 'roller_skate', 'Rollerblade', 'root_beer', 'safety_pin', 'salad_plate', 'salmon_(food)', 'satchel', 'saucepan', 'sawhorse', 'saxophone', 'scarecrow', 'scraper', 'seaplane', 'sharpener', 'Sharpie', 'shaver_(electric)', 'shawl', 'shears', 'shepherd_dog', 'sherbert', 'shot_glass', 'shower_cap', 'shredder_(for_paper)', 'skullcap', 'sling_(bandage)', 'smoothie', 'snake', 'softball', 'sombrero', 'soup_bowl', 'soya_milk', 'space_shuttle', 'sparkler_(fireworks)', 'spear', 'crawfish', 'squid_(food)', 'stagecoach', 'steak_knife', 'stepladder', 'stew', 'stirrer', 'string_cheese', 'stylus', 'subwoofer', 'sugar_bowl', 'sugarcane_(plant)', 'syringe', 'Tabasco_sauce', 'table-tennis_table', 'tachometer', 'taco', 'tambourine', 'army_tank', 'telephoto_lens', 'tequila', 'thimble', 'trampoline', 'trench_coat', 'triangle_(musical_instrument)', 'truffle_(chocolate)', 'vat', 'turnip', 'unicycle', 'vinegar', 'violin', 'vodka', 'vulture', 'waffle_iron', 'walrus', 'wardrobe', 'washbasin', 'water_heater', 'water_gun', 'wolf'
])

COCO_ASD_FEW_SHOT_CLASSES_65_8_7_set1 = np.array([
    'train', 'parking meter', 'bear', 'snowboard', 'fork', 'hot dog', 'toaster', 'hair drier'
])
COCO_ASD_FEW_SHOT_CLASSES_65_8_7_set2 = np.array([
    'train', 'parking meter', 'cat', 'bear', 'snowboard', 'fork', 'toilet', 'toaster'
])
COCO_ASD_FEW_SHOT_CLASSES_65_8_7_set3 = np.array([
    'train', 'parking meter', 'cat', 'bear', 'snowboard', 'fork', 'mouse', 'toaster'
])

COCO_ASD_ZERO_SHOT_CLASSES_65_8_7_set1 = np.array([
    'airplane', 'cat', 'suitcase', 'frisbee', 'sandwich', 'toilet', 'mouse'
])
COCO_ASD_ZERO_SHOT_CLASSES_65_8_7_set2 = np.array([
    'airplane',  'suitcase', 'frisbee', 'sandwich', 'hot dog', 'mouse', 'hair drier'
])
COCO_ASD_ZERO_SHOT_CLASSES_65_8_7_set3 = np.array([
    'airplane',  'suitcase', 'frisbee',  'sandwich', 'hot dog', 'toilet', 'hair drier'
])

COCO_ASD_FEW_ZERO_SHOT_CLASSES_65_8_7 = np.array([
    'airplane', 'train', 'parking meter', 'cat', 'bear', 'suitcase', 'frisbee', 'snowboard', 'fork',
               'sandwich', 'hot dog', 'toilet', 'mouse', 'toaster', 'hair drier'
])

VOC_UNSEEN_CLASSES = np.array(['car', 'dog', 'sofa', 'train'])

COCO_SEEN_CLASSES_48_17 = np.array([
    "toilet",
    "bicycle",
    "apple",
    "train",
    "laptop",
    "carrot",
    "motorcycle",
    "oven",
    "chair",
    "mouse",
    "boat",
    "kite",
    "sheep",
    "horse",
    "sandwich",
    "clock",
    "tv",
    "backpack",
    "toaster",
    "bowl",
    "microwave",
    "bench",
    "book",
    "orange",
    "bird",
    "pizza",
    "fork",
    "frisbee",
    "bear",
    "vase",
    "toothbrush",
    "spoon",
    "giraffe",
    "handbag",
    "broccoli",
    "refrigerator",
    "remote",
    "surfboard",
    "car",
    "bed",
    "banana",
    "donut",
    "skis",
    "person",
    "truck",
    "bottle",
    "suitcase",
    "zebra",
])

COCO_UNSEEN_CLASSES_48_17 = np.array([
    "umbrella",
    "cow",
    "cup",
    "bus",
    "keyboard",
    "skateboard",
    "dog",
    "couch",
    "tie",
    "snowboard",
    "sink",
    "elephant",
    "cake",
    "scissors",
    "airplane",
    "cat",
    "knife"
])

COCO_SEEN_CLASSES_60_20 = np.array([
    'truck', 'traffic light', 'fire hydrant', 'stop sign',
    'parking meter', 'bench', 'elephant', 'bear', 'zebra',
    'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
    'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard',
    'surfboard', 'tennis racket', 'wine glass', 'cup', 'fork',
    'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
    'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut',
    'cake', 'bed', 'toilet', 'laptop', 'mouse', 'remote',
    'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
    'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
    'teddy bear', 'hair drier', 'toothbrush'
])

COCO_UNSEEN_CLASSES_60_20 = np.array([
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'boat', 'bird', 'cat', 'dog', 'horse', 'sheep',
    'cow', 'bottle', 'chair', 'couch', 'potted plant',
    'dining table', 'tv'
])

def get_class_labels(dataset):
    if dataset == 'coco':
        return COCO_ALL_CLASSES
    elif dataset == 'voc': 
        return VOC_ALL_CLASSES
    elif dataset == 'imagenet':
        return IMAGENET_ALL_CLASSES


def get_unseen_class_labels(dataset, split='65_15'):
    if dataset == 'coco':
        if split=='65_15':
            return COCO_UNSEEN_CLASSES_65_15
        elif split=='60_20':
            return COCO_UNSEEN_CLASSES_60_20
        else:
            return COCO_UNSEEN_CLASSES_48_17
    elif dataset == 'voc': 
        return VOC_UNSEEN_CLASSES
    elif dataset == 'imagenet':
        return IMAGENET_UNSEEN_CLASSES
    elif dataset == 'lvis':
        return LVIS_NOVEL_CLASSES

def get_unseen_class_ids(dataset, split='65_15'):
    if dataset == 'coco':
        return get_unseen_coco_cat_ids(split)
    elif dataset == 'voc':
        return get_unseen_voc_ids()
    elif dataset == 'imagenet':
        return get_unseen_imagenet_ids()
    elif dataset == 'lvis':
        return get_unseen_lvis_ids()

# asd
# zero_shot
def get_asd_zero_shot_class_labels(dataset, split='65_8_7', fs_set=1):
    if dataset == 'coco':
        if fs_set == 1:
            return COCO_ASD_ZERO_SHOT_CLASSES_65_8_7_set1 if '65' in split else COCO_UNSEEN_CLASSES_48_17
        elif fs_set == 2: 
            return COCO_ASD_ZERO_SHOT_CLASSES_65_8_7_set2 if '65' in split else COCO_UNSEEN_CLASSES_48_17
        elif fs_set == 3:
            return COCO_ASD_ZERO_SHOT_CLASSES_65_8_7_set3 if '65' in split else COCO_UNSEEN_CLASSES_48_17

def get_asd_zero_shot_class_ids(dataset, split='65_8_7', fs_set=1):
    if dataset == 'coco':
        return get_asd_zero_shot_coco_cat_ids(split, fs_set=fs_set)

def get_asd_zero_shot_coco_cat_ids(split='65_8_7', fs_set=1):
    if fs_set == 1:
        ZERO_SHOT_CLASSES = COCO_ASD_ZERO_SHOT_CLASSES_65_8_7_set1 if '65' in split else COCO_UNSEEN_CLASSES_48_17
    elif fs_set == 2:
        ZERO_SHOT_CLASSES = COCO_ASD_ZERO_SHOT_CLASSES_65_8_7_set2 if '65' in split else COCO_UNSEEN_CLASSES_48_17
    elif fs_set == 3:
        ZERO_SHOT_CLASSES = COCO_ASD_ZERO_SHOT_CLASSES_65_8_7_set3 if '65' in split else COCO_UNSEEN_CLASSES_48_17
    ids = np.where(np.isin(COCO_ALL_CLASSES, ZERO_SHOT_CLASSES))[0]
    return ids

def get_asd_few_zero_shot_class_labels(dataset, split='65_8_7'):
    if dataset == 'coco':
        if '65' in split:
            return COCO_ASD_FEW_ZERO_SHOT_CLASSES_65_8_7
        elif '60' in split:
            return COCO_UNSEEN_CLASSES_60_20
        else:
            return COCO_UNSEEN_CLASSES_48_17
    elif dataset == 'voc' and split=='16_4':
        return VOC_UNSEEN_CLASSES
    elif dataset == 'voc' and split=='15_5_1':
        return VOC_NOVEL_CLASSES_SPLIT1
    elif dataset == 'voc' and split=='15_5_2':
        return VOC_NOVEL_CLASSES_SPLIT2
    elif dataset == 'voc' and split=='15_5_3':
        return VOC_NOVEL_CLASSES_SPLIT3
    elif dataset == 'lvis':
        return LVIS_NOVEL_CLASSES

def get_asd_few_zero_shot_class_ids(dataset, split='65_8_7'):
    if dataset == 'coco':
        if '65' in split:
            FEW_ZERO_SHOT_CLASSES = COCO_ASD_FEW_ZERO_SHOT_CLASSES_65_8_7 
        elif '60' in split:
            FEW_ZERO_SHOT_CLASSES = COCO_UNSEEN_CLASSES_60_20
        else:
            FEW_ZERO_SHOT_CLASSES = COCO_UNSEEN_CLASSES_48_17

        ids = np.where(np.isin(COCO_ALL_CLASSES, FEW_ZERO_SHOT_CLASSES))[0]
        return ids
    elif dataset == 'voc' and split=='16_4':
        return get_unseen_voc_ids()
    elif dataset == 'voc' and split=='15_5_1':
        ids = np.where(np.isin(VOC_ALL_CLASSES_SPLIT1, VOC_NOVEL_CLASSES_SPLIT1))[0]
        return ids
    elif dataset == 'voc' and split=='15_5_2':
        ids = np.where(np.isin(VOC_ALL_CLASSES_SPLIT2, VOC_NOVEL_CLASSES_SPLIT2))[0]
        return ids
    elif dataset == 'voc' and split=='15_5_3':
        ids = np.where(np.isin(VOC_ALL_CLASSES_SPLIT3, VOC_NOVEL_CLASSES_SPLIT3))[0]
        return ids
    elif dataset == 'lvis':
        ids = np.where(np.isin(LVIS_ALL_CLASSES, LVIS_NOVEL_CLASSES))[0]
        return ids

def get_asd_few_zero_shot_coco_cat_ids(split='65_8_7'):
    if '65' in split:
        FEW_ZERO_SHOT_CLASSES = COCO_ASD_FEW_ZERO_SHOT_CLASSES_65_8_7
    elif '60' in split:
        FEW_ZERO_SHOT_CLASSES = COCO_UNSEEN_CLASSES_60_20
    else:
        FEW_ZERO_SHOT_CLASSES = COCO_UNSEEN_CLASSES_48_17
    
    ids = np.where(np.isin(COCO_ALL_CLASSES, FEW_ZERO_SHOT_CLASSES))[0]
    return ids

def get_asd_base_few_zero_shot_class_labels(dataset, split='65_8_7'):
    if dataset == 'coco':
        if '65' in split:
            return COCO_ALL_CLASSES  
        elif '60' in split:
            return COCO_UNSEEN_CLASSES_60_20
        else:
            return COCO_UNSEEN_CLASSES_48_17
    elif dataset == 'lvis':
        return LVIS_ALL_CLASSES

def get_asd_base_few_zero_shot_class_ids(dataset, split='65_8_7'):
    if dataset == 'coco':
        if '65' in split:
            BASE_FEW_ZERO_SHOT_CLASSES = COCO_ALL_CLASSES  
        elif '60' in split:
            BASE_FEW_ZERO_SHOT_CLASSES = COCO_ALL_CLASSES
        else:
            BASE_FEW_ZERO_SHOT_CLASSES = COCO_UNSEEN_CLASSES_48_17
        ids = np.where(np.isin(COCO_ALL_CLASSES, BASE_FEW_ZERO_SHOT_CLASSES))[0]
        return ids
    elif dataset == 'lvis':
        ids = np.where(np.isin(LVIS_ALL_CLASSES, LVIS_ALL_CLASSES))[0]
        return ids

def get_seen_class_ids(dataset, split='65_15'):
    if dataset == 'coco':
        return get_seen_coco_cat_ids(split)
    elif dataset == 'voc' and split=='16_4':
        return get_seen_voc_ids()
    elif dataset == 'voc' and split=='15_5_1':
        ids = np.where(np.isin(VOC_ALL_CLASSES_SPLIT1, VOC_BASE_CLASSES_SPLIT1))[0] 
        return ids
    elif dataset == 'voc' and split=='15_5_2':
        ids = np.where(np.isin(VOC_ALL_CLASSES_SPLIT2, VOC_BASE_CLASSES_SPLIT2))[0] 
        return ids
    elif dataset == 'voc' and split=='15_5_3':
        ids = np.where(np.isin(VOC_ALL_CLASSES_SPLIT3, VOC_BASE_CLASSES_SPLIT3))[0] 
        return ids
    elif dataset == 'imagenet':
        return get_seen_imagenet_ids()
    elif dataset == 'lvis':
        return get_seen_lvis_ids()

def get_unseen_coco_cat_ids(split='65_15'):
    if split=='65_15':
        UNSEEN_CLASSES = COCO_UNSEEN_CLASSES_65_15 
    elif split=='60_20':
        UNSEEN_CLASSES = COCO_UNSEEN_CLASSES_60_20
    else: 
        UNSEEN_CLASSES = COCO_UNSEEN_CLASSES_48_17
    ids = np.where(np.isin(COCO_ALL_CLASSES, UNSEEN_CLASSES))[0]
    return ids

def get_seen_coco_cat_ids(split='65_15'):
    if split=='65_15':
        seen_classes = np.setdiff1d(COCO_ALL_CLASSES, COCO_UNSEEN_CLASSES_65_15)
    elif split=='60_20':
        seen_classes = np.setdiff1d(COCO_ALL_CLASSES, COCO_UNSEEN_CLASSES_60_20)
    else:
        seen_classes = COCO_SEEN_CLASSES_48_17
    ids = np.where(np.isin(COCO_ALL_CLASSES, seen_classes))[0]
    return ids

def get_unseen_lvis_ids():
    ids = np.where(np.isin(LVIS_ALL_CLASSES, LVIS_NOVEL_CLASSES))[0]
    return ids

def get_seen_lvis_ids():
    ids = np.where(np.isin(LVIS_ALL_CLASSES, LVIS_BASE_CLASSES))[0]
    return ids

def get_unseen_voc_ids():
    ids = np.where(np.isin(VOC_ALL_CLASSES, VOC_UNSEEN_CLASSES))[0]
    # ids = np.concatenate(([0], ids+1))
    return ids

def get_seen_voc_ids():
    seen_classes = np.setdiff1d(VOC_ALL_CLASSES, VOC_UNSEEN_CLASSES)
    ids = np.where(np.isin(VOC_ALL_CLASSES, seen_classes))[0] 
    # ids = np.concatenate(([0], ids+1))
    return ids

def get_unseen_imagenet_ids():
    ids = np.where(np.isin(IMAGENET_ALL_CLASSES, IMAGENET_UNSEEN_CLASSES))[0] + 1
    return ids

def get_seen_imagenet_ids():
    seen_classes = np.setdiff1d(IMAGENET_ALL_CLASSES, IMAGENET_UNSEEN_CLASSES)
    ids = np.where(np.isin(IMAGENET_ALL_CLASSES, seen_classes))[0]
    return ids
