# Copyright (c) OpenMMLab. All rights reserved.
import itertools
import logging
import os.path as osp
from collections import OrderedDict
from typing import Dict, List, Optional, Sequence, Union

import mmcv
import numpy as np
from mmcv.utils import print_log
from mmdet.datasets.api_wrappers import COCO, COCOeval
from lvis import LVIS, LVISResults, LVISEval
from mmdet.datasets.builder import DATASETS
from mmdet.datasets.coco import CocoDataset
from mmdet.datasets.lvis import LVISV05Dataset, LVISV1Dataset
from terminaltables import AsciiTable

from .base import BaseFewShotDataset
import ipdb 
# pre-defined classes split for any shot setting
LVIS_SPLIT = dict(
    ALL_CLASSES=('aerosol_can', 'air_conditioner', 'airplane', 'alarm_clock', 'alcohol', 'alligator', 'almond', 'ambulance', 'amplifier', 'anklet', 'antenna', 'apple', 'applesauce', 'apricot', 'apron', 'aquarium', 'arctic_(type_of_shoe)', 'armband', 'armchair', 'armoire', 'armor', 'artichoke', 'trash_can', 'ashtray', 'asparagus', 'atomizer', 'avocado', 'award', 'awning', 'ax', 'baboon', 'baby_buggy', 'basketball_backboard', 'backpack', 'handbag', 'suitcase', 'bagel', 'bagpipe', 'baguet', 'bait', 'ball', 'ballet_skirt', 'balloon', 'bamboo', 'banana', 'Band_Aid', 'bandage', 'bandanna', 'banjo', 'banner', 'barbell', 'barge', 'barrel', 'barrette', 'barrow', 'baseball_base', 'baseball', 'baseball_bat', 'baseball_cap', 'baseball_glove', 'basket', 'basketball', 'bass_horn', 'bat_(animal)', 'bath_mat', 'bath_towel', 'bathrobe', 'bathtub', 'batter_(food)', 'battery', 'beachball', 'bead', 'bean_curd', 'beanbag', 'beanie', 'bear', 'bed', 'bedpan', 'bedspread', 'cow', 'beef_(food)', 'beeper', 'beer_bottle', 'beer_can', 'beetle', 'bell', 'bell_pepper', 'belt', 'belt_buckle', 'bench', 'beret', 'bib', 'Bible', 'bicycle', 'visor', 'billboard', 'binder', 'binoculars', 'bird', 'birdfeeder', 'birdbath', 'birdcage', 'birdhouse', 'birthday_cake', 'birthday_card', 'pirate_flag', 'black_sheep', 'blackberry', 'blackboard', 'blanket', 'blazer', 'blender', 'blimp', 'blinker', 'blouse', 'blueberry', 'gameboard', 'boat', 'bob', 'bobbin', 'bobby_pin', 'boiled_egg', 'bolo_tie', 'deadbolt', 'bolt', 'bonnet', 'book', 'bookcase', 'booklet', 'bookmark', 'boom_microphone', 'boot', 'bottle', 'bottle_opener', 'bouquet', 'bow_(weapon)', 'bow_(decorative_ribbons)', 'bow-tie', 'bowl', 'pipe_bowl', 'bowler_hat', 'bowling_ball', 'box', 'boxing_glove', 'suspenders', 'bracelet', 'brass_plaque', 'brassiere', 'bread-bin', 'bread', 'breechcloth', 'bridal_gown', 'briefcase', 'broccoli', 'broach', 'broom', 'brownie', 'brussels_sprouts', 'bubble_gum', 'bucket', 'horse_buggy', 'bull', 'bulldog', 'bulldozer', 'bullet_train', 'bulletin_board', 'bulletproof_vest', 'bullhorn', 'bun', 'bunk_bed', 'buoy', 'burrito', 'bus_(vehicle)', 'business_card', 'butter', 'butterfly', 'button', 'cab_(taxi)', 'cabana', 'cabin_car', 'cabinet', 'locker', 'cake', 'calculator', 'calendar', 'calf', 'camcorder', 'camel', 'camera', 'camera_lens', 'camper_(vehicle)', 'can', 'can_opener', 'candle', 'candle_holder', 'candy_bar', 'candy_cane', 'walking_cane', 'canister', 'canoe', 'cantaloup', 'canteen', 'cap_(headwear)', 'bottle_cap', 'cape', 'cappuccino', 'car_(automobile)', 'railcar_(part_of_a_train)', 'elevator_car', 'car_battery', 'identity_card', 'card', 'cardigan', 'cargo_ship', 'carnation', 'horse_carriage', 'carrot', 'tote_bag', 'cart', 'carton', 'cash_register', 'casserole', 'cassette', 'cast', 'cat', 'cauliflower', 'cayenne_(spice)', 'CD_player', 'celery', 'cellular_telephone', 'chain_mail', 'chair', 'chaise_longue', 'chalice', 'chandelier', 'chap', 'checkbook', 'checkerboard', 'cherry', 'chessboard', 'chicken_(animal)', 'chickpea', 'chili_(vegetable)', 'chime', 'chinaware', 'crisp_(potato_chip)', 'poker_chip', 'chocolate_bar', 'chocolate_cake', 'chocolate_milk', 'chocolate_mousse', 'choker', 'chopping_board', 'chopstick', 'Christmas_tree', 'slide', 'cider', 'cigar_box', 'cigarette', 'cigarette_case', 'cistern', 'clarinet', 'clasp', 'cleansing_agent', 'cleat_(for_securing_rope)', 'clementine', 'clip', 'clipboard', 'clippers_(for_plants)', 'cloak', 'clock', 'clock_tower', 'clothes_hamper', 'clothespin', 'clutch_bag', 'coaster', 'coat', 'coat_hanger', 'coatrack', 'cock', 'cockroach', 'cocoa_(beverage)', 'coconut', 'coffee_maker', 'coffee_table', 'coffeepot', 'coil', 'coin', 'colander', 'coleslaw', 'coloring_material', 'combination_lock', 'pacifier', 'comic_book', 'compass', 'computer_keyboard', 'condiment', 'cone', 'control', 'convertible_(automobile)', 'sofa_bed', 'cooker', 'cookie', 'cooking_utensil', 'cooler_(for_food)', 'cork_(bottle_plug)', 'corkboard', 'corkscrew', 'edible_corn', 'cornbread', 'cornet', 'cornice', 'cornmeal', 'corset', 'costume', 'cougar', 'coverall', 'cowbell', 'cowboy_hat', 'crab_(animal)', 'crabmeat', 'cracker', 'crape', 'crate', 'crayon', 'cream_pitcher', 'crescent_roll', 'crib', 'crock_pot', 'crossbar', 'crouton', 'crow', 'crowbar', 'crown', 'crucifix', 'cruise_ship', 'police_cruiser', 'crumb', 'crutch', 'cub_(animal)', 'cube', 'cucumber', 'cufflink', 'cup', 'trophy_cup', 'cupboard', 'cupcake', 'hair_curler', 'curling_iron', 'curtain', 'cushion', 'cylinder', 'cymbal', 'dagger', 'dalmatian', 'dartboard', 'date_(fruit)', 'deck_chair', 'deer', 'dental_floss', 'desk', 'detergent', 'diaper', 'diary', 'die', 'dinghy', 'dining_table', 'tux', 'dish', 'dish_antenna', 'dishrag', 'dishtowel', 'dishwasher', 'dishwasher_detergent', 'dispenser', 'diving_board', 'Dixie_cup', 'dog', 'dog_collar', 'doll', 'dollar', 'dollhouse', 'dolphin', 'domestic_ass', 'doorknob', 'doormat', 'doughnut', 'dove', 'dragonfly', 'drawer', 'underdrawers', 'dress', 'dress_hat', 'dress_suit', 'dresser', 'drill', 'drone', 'dropper', 'drum_(musical_instrument)', 'drumstick', 'duck', 'duckling', 'duct_tape', 'duffel_bag', 'dumbbell', 'dumpster', 'dustpan', 'eagle', 'earphone', 'earplug', 'earring', 'easel', 'eclair', 'eel', 'egg', 'egg_roll', 'egg_yolk', 'eggbeater', 'eggplant', 'electric_chair', 'refrigerator', 'elephant', 'elk', 'envelope', 'eraser', 'escargot', 'eyepatch', 'falcon', 'fan', 'faucet', 'fedora', 'ferret', 'Ferris_wheel', 'ferry', 'fig_(fruit)', 'fighter_jet', 'figurine', 'file_cabinet', 'file_(tool)', 'fire_alarm', 'fire_engine', 'fire_extinguisher', 'fire_hose', 'fireplace', 'fireplug', 'first-aid_kit', 'fish', 'fish_(food)', 'fishbowl', 'fishing_rod', 'flag', 'flagpole', 'flamingo', 'flannel', 'flap', 'flash', 'flashlight', 'fleece', 'flip-flop_(sandal)', 'flipper_(footwear)', 'flower_arrangement', 'flute_glass', 'foal', 'folding_chair', 'food_processor', 'football_(American)', 'football_helmet', 'footstool', 'fork', 'forklift', 'freight_car', 'French_toast', 'freshener', 'frisbee', 'frog', 'fruit_juice', 'frying_pan', 'fudge', 'funnel', 'futon', 'gag', 'garbage', 'garbage_truck', 'garden_hose', 'gargle', 'gargoyle', 'garlic', 'gasmask', 'gazelle', 'gelatin', 'gemstone', 'generator', 'giant_panda', 'gift_wrap', 'ginger', 'giraffe', 'cincture', 'glass_(drink_container)', 'globe', 'glove', 'goat', 'goggles', 'goldfish', 'golf_club', 'golfcart', 'gondola_(boat)', 'goose', 'gorilla', 'gourd', 'grape', 'grater', 'gravestone', 'gravy_boat', 'green_bean', 'green_onion', 'griddle', 'grill', 'grits', 'grizzly', 'grocery_bag', 'guitar', 'gull', 'gun', 'hairbrush', 'hairnet', 'hairpin', 'halter_top', 'ham', 'hamburger', 'hammer', 'hammock', 'hamper', 'hamster', 'hair_dryer', 'hand_glass', 'hand_towel', 'handcart', 'handcuff', 'handkerchief', 'handle', 'handsaw', 'hardback_book', 'harmonium', 'hat', 'hatbox', 'veil', 'headband', 'headboard', 'headlight', 'headscarf', 'headset', 'headstall_(for_horses)', 'heart', 'heater', 'helicopter', 'helmet', 'heron', 'highchair', 'hinge', 'hippopotamus', 'hockey_stick', 'hog', 'home_plate_(baseball)', 'honey', 'fume_hood', 'hook', 'hookah', 'hornet', 'horse', 'hose', 'hot-air_balloon', 'hotplate', 'hot_sauce', 'hourglass', 'houseboat', 'hummingbird', 'hummus', 'polar_bear', 'icecream', 'popsicle', 'ice_maker', 'ice_pack', 'ice_skate', 'igniter', 'inhaler', 'iPod', 'iron_(for_clothing)', 'ironing_board', 'jacket', 'jam', 'jar', 'jean', 'jeep', 'jelly_bean', 'jersey', 'jet_plane', 'jewel', 'jewelry', 'joystick', 'jumpsuit', 'kayak', 'keg', 'kennel', 'kettle', 'key', 'keycard', 'kilt', 'kimono', 'kitchen_sink', 'kitchen_table', 'kite', 'kitten', 'kiwi_fruit', 'knee_pad', 'knife', 'knitting_needle', 'knob', 'knocker_(on_a_door)', 'koala', 'lab_coat', 'ladder', 'ladle', 'ladybug', 'lamb_(animal)', 'lamb-chop', 'lamp', 'lamppost', 'lampshade', 'lantern', 'lanyard', 'laptop_computer', 'lasagna', 'latch', 'lawn_mower', 'leather', 'legging_(clothing)', 'Lego', 'legume', 'lemon', 'lemonade', 'lettuce', 'license_plate', 'life_buoy', 'life_jacket', 'lightbulb', 'lightning_rod', 'lime', 'limousine', 'lion', 'lip_balm', 'liquor', 'lizard', 'log', 'lollipop', 'speaker_(stero_equipment)', 'loveseat', 'machine_gun', 'magazine', 'magnet', 'mail_slot', 'mailbox_(at_home)', 'mallard', 'mallet', 'mammoth', 'manatee', 'mandarin_orange', 'manger', 'manhole', 'map', 'marker', 'martini', 'mascot', 'mashed_potato', 'masher', 'mask', 'mast', 'mat_(gym_equipment)', 'matchbox', 'mattress', 'measuring_cup', 'measuring_stick', 'meatball', 'medicine', 'melon', 'microphone', 'microscope', 'microwave_oven', 'milestone', 'milk', 'milk_can', 'milkshake', 'minivan', 'mint_candy', 'mirror', 'mitten', 'mixer_(kitchen_tool)', 'money', 'monitor_(computer_equipment) computer_monitor', 'monkey', 'motor', 'motor_scooter', 'motor_vehicle', 'motorcycle', 'mound_(baseball)', 'mouse_(computer_equipment)', 'mousepad', 'muffin', 'mug', 'mushroom', 'music_stool', 'musical_instrument', 'nailfile', 'napkin', 'neckerchief', 'necklace', 'necktie', 'needle', 'nest', 'newspaper', 'newsstand', 'nightshirt', 'nosebag_(for_animals)', 'noseband_(for_animals)', 'notebook', 'notepad', 'nut', 'nutcracker', 'oar', 'octopus_(food)', 'octopus_(animal)', 'oil_lamp', 'olive_oil', 'omelet', 'onion', 'orange_(fruit)', 'orange_juice', 'ostrich', 'ottoman', 'oven', 'overalls_(clothing)', 'owl', 'packet', 'inkpad', 'pad', 'paddle', 'padlock', 'paintbrush', 'painting', 'pajamas', 'palette', 'pan_(for_cooking)', 'pan_(metal_container)', 'pancake', 'pantyhose', 'papaya', 'paper_plate', 'paper_towel', 'paperback_book', 'paperweight', 'parachute', 'parakeet', 'parasail_(sports)', 'parasol', 'parchment', 'parka', 'parking_meter', 'parrot', 'passenger_car_(part_of_a_train)', 'passenger_ship', 'passport', 'pastry', 'patty_(food)', 'pea_(food)', 'peach', 'peanut_butter', 'pear', 'peeler_(tool_for_fruit_and_vegetables)', 'wooden_leg', 'pegboard', 'pelican', 'pen', 'pencil', 'pencil_box', 'pencil_sharpener', 'pendulum', 'penguin', 'pennant', 'penny_(coin)', 'pepper', 'pepper_mill', 'perfume', 'persimmon', 'person', 'pet', 'pew_(church_bench)', 'phonebook', 'phonograph_record', 'piano', 'pickle', 'pickup_truck', 'pie', 'pigeon', 'piggy_bank', 'pillow', 'pin_(non_jewelry)', 'pineapple', 'pinecone', 'ping-pong_ball', 'pinwheel', 'tobacco_pipe', 'pipe', 'pistol', 'pita_(bread)', 'pitcher_(vessel_for_liquid)', 'pitchfork', 'pizza', 'place_mat', 'plate', 'platter', 'playpen', 'pliers', 'plow_(farm_equipment)', 'plume', 'pocket_watch', 'pocketknife', 'poker_(fire_stirring_tool)', 'pole', 'polo_shirt', 'poncho', 'pony', 'pool_table', 'pop_(soda)', 'postbox_(public)', 'postcard', 'poster', 'pot', 'flowerpot', 'potato', 'potholder', 'pottery', 'pouch', 'power_shovel', 'prawn', 'pretzel', 'printer', 'projectile_(weapon)', 'projector', 'propeller', 'prune', 'pudding', 'puffer_(fish)', 'puffin', 'pug-dog', 'pumpkin', 'puncher', 'puppet', 'puppy', 'quesadilla', 'quiche', 'quilt', 'rabbit', 'race_car', 'racket', 'radar', 'radiator', 'radio_receiver', 'radish', 'raft', 'rag_doll', 'raincoat', 'ram_(animal)', 'raspberry', 'rat', 'razorblade', 'reamer_(juicer)', 'rearview_mirror', 'receipt', 'recliner', 'record_player', 'reflector', 'remote_control', 'rhinoceros', 'rib_(food)', 'rifle', 'ring', 'river_boat', 'road_map', 'robe', 'rocking_chair', 'rodent', 'roller_skate', 'Rollerblade', 'rolling_pin', 'root_beer', 'router_(computer_equipment)', 'rubber_band', 'runner_(carpet)', 'plastic_bag', 'saddle_(on_an_animal)', 'saddle_blanket', 'saddlebag', 'safety_pin', 'sail', 'salad', 'salad_plate', 'salami', 'salmon_(fish)', 'salmon_(food)', 'salsa', 'saltshaker', 'sandal_(type_of_shoe)', 'sandwich', 'satchel', 'saucepan', 'saucer', 'sausage', 'sawhorse', 'saxophone', 'scale_(measuring_instrument)', 'scarecrow', 'scarf', 'school_bus', 'scissors', 'scoreboard', 'scraper', 'screwdriver', 'scrubbing_brush', 'sculpture', 'seabird', 'seahorse', 'seaplane', 'seashell', 'sewing_machine', 'shaker', 'shampoo', 'shark', 'sharpener', 'Sharpie', 'shaver_(electric)', 'shaving_cream', 'shawl', 'shears', 'sheep', 'shepherd_dog', 'sherbert', 'shield', 'shirt', 'shoe', 'shopping_bag', 'shopping_cart', 'short_pants', 'shot_glass', 'shoulder_bag', 'shovel', 'shower_head', 'shower_cap', 'shower_curtain', 'shredder_(for_paper)', 'signboard', 'silo', 'sink', 'skateboard', 'skewer', 'ski', 'ski_boot', 'ski_parka', 'ski_pole', 'skirt', 'skullcap', 'sled', 'sleeping_bag', 'sling_(bandage)', 'slipper_(footwear)', 'smoothie', 'snake', 'snowboard', 'snowman', 'snowmobile', 'soap', 'soccer_ball', 'sock', 'sofa', 'softball', 'solar_array', 'sombrero', 'soup', 'soup_bowl', 'soupspoon', 'sour_cream', 'soya_milk', 'space_shuttle', 'sparkler_(fireworks)', 'spatula', 'spear', 'spectacles', 'spice_rack', 'spider', 'crawfish', 'sponge', 'spoon', 'sportswear', 'spotlight', 'squid_(food)', 'squirrel', 'stagecoach', 'stapler_(stapling_machine)', 'starfish', 'statue_(sculpture)', 'steak_(food)', 'steak_knife', 'steering_wheel', 'stepladder', 'step_stool', 'stereo_(sound_system)', 'stew', 'stirrer', 'stirrup', 'stool', 'stop_sign', 'brake_light', 'stove', 'strainer', 'strap', 'straw_(for_drinking)', 'strawberry', 'street_sign', 'streetlight', 'string_cheese', 'stylus', 'subwoofer', 'sugar_bowl', 'sugarcane_(plant)', 'suit_(clothing)', 'sunflower', 'sunglasses', 'sunhat', 'surfboard', 'sushi', 'mop', 'sweat_pants', 'sweatband', 'sweater', 'sweatshirt', 'sweet_potato', 'swimsuit', 'sword', 'syringe', 'Tabasco_sauce', 'table-tennis_table', 'table', 'table_lamp', 'tablecloth', 'tachometer', 'taco', 'tag', 'taillight', 'tambourine', 'army_tank', 'tank_(storage_vessel)', 'tank_top_(clothing)', 'tape_(sticky_cloth_or_paper)', 'tape_measure', 'tapestry', 'tarp', 'tartan', 'tassel', 'tea_bag', 'teacup', 'teakettle', 'teapot', 'teddy_bear', 'telephone', 'telephone_booth', 'telephone_pole', 'telephoto_lens', 'television_camera', 'television_set', 'tennis_ball', 'tennis_racket', 'tequila', 'thermometer', 'thermos_bottle', 'thermostat', 'thimble', 'thread', 'thumbtack', 'tiara', 'tiger', 'tights_(clothing)', 'timer', 'tinfoil', 'tinsel', 'tissue_paper', 'toast_(food)', 'toaster', 'toaster_oven', 'toilet', 'toilet_tissue', 'tomato', 'tongs', 'toolbox', 'toothbrush', 'toothpaste', 'toothpick', 'cover', 'tortilla', 'tow_truck', 'towel', 'towel_rack', 'toy', 'tractor_(farm_equipment)', 'traffic_light', 'dirt_bike', 'trailer_truck', 'train_(railroad_vehicle)', 'trampoline', 'tray', 'trench_coat', 'triangle_(musical_instrument)', 'tricycle', 'tripod', 'trousers', 'truck', 'truffle_(chocolate)', 'trunk', 'vat', 'turban', 'turkey_(food)', 'turnip', 'turtle', 'turtleneck_(clothing)', 'typewriter', 'umbrella', 'underwear', 'unicycle', 'urinal', 'urn', 'vacuum_cleaner', 'vase', 'vending_machine', 'vent', 'vest', 'videotape', 'vinegar', 'violin', 'vodka', 'volleyball', 'vulture', 'waffle', 'waffle_iron', 'wagon', 'wagon_wheel', 'walking_stick', 'wall_clock', 'wall_socket', 'wallet', 'walrus', 'wardrobe', 'washbasin', 'automatic_washer', 'watch', 'water_bottle', 'water_cooler', 'water_faucet', 'water_heater', 'water_jug', 'water_gun', 'water_scooter', 'water_ski', 'water_tower', 'watering_can', 'watermelon', 'weathervane', 'webcam', 'wedding_cake', 'wedding_ring', 'wet_suit', 'wheel', 'wheelchair', 'whipped_cream', 'whistle', 'wig', 'wind_chime', 'windmill', 'window_box_(for_plants)', 'windshield_wiper', 'windsock', 'wine_bottle', 'wine_bucket', 'wineglass', 'blinder_(for_horses)', 'wok', 'wolf', 'wooden_spoon', 'wreath', 'wrench', 'wristband', 'wristlet', 'yacht', 'yogurt', 'yoke_(animal_equipment)', 'zebra', 'zucchini'),
    NOVEL_CLASSES=('applesauce', 'apricot', 'arctic_(type_of_shoe)', 'armoire', 'armor', 'ax', 'baboon', 'bagpipe', 'baguet', 'bait', 'ballet_skirt', 'banjo', 'barbell', 'barge', 'bass_horn', 'batter_(food)', 'beachball', 'bedpan', 'beeper', 'beetle', 'Bible', 'birthday_card', 'pirate_flag', 'blimp', 'gameboard', 'bob', 'bolo_tie', 'bonnet', 'bookmark', 'boom_microphone', 'bow_(weapon)', 'pipe_bowl', 'bowling_ball', 'boxing_glove', 'brass_plaque', 'breechcloth', 'broach', 'bubble_gum', 'horse_buggy', 'bulldozer', 'bulletproof_vest', 'burrito', 'cabana', 'locker', 'candy_bar', 'canteen', 'elevator_car', 'car_battery', 'cargo_ship', 'carnation', 'casserole', 'cassette', 'chain_mail', 'chaise_longue', 'chalice', 'chap', 'checkbook', 'checkerboard', 'chessboard', 'chime', 'chinaware', 'poker_chip', 'chocolate_milk', 'chocolate_mousse', 'cider', 'cigar_box', 'clarinet', 'cleat_(for_securing_rope)', 'clementine', 'clippers_(for_plants)', 'cloak', 'clutch_bag', 'cockroach', 'cocoa_(beverage)', 'coil', 'coloring_material', 'combination_lock', 'comic_book', 'compass', 'convertible_(automobile)', 'sofa_bed', 'cooker', 'cooking_utensil', 'corkboard', 'cornbread', 'cornmeal', 'cougar', 'coverall', 'crabmeat', 'crape', 'cream_pitcher', 'crouton', 'crowbar', 'hair_curler', 'curling_iron', 'cylinder', 'cymbal', 'dagger', 'dalmatian', 'date_(fruit)', 'detergent', 'diary', 'die', 'dinghy', 'tux', 'dishwasher_detergent', 'diving_board', 'dollar', 'dollhouse', 'dove', 'dragonfly', 'drone', 'dropper', 'drumstick', 'dumbbell', 'dustpan', 'earplug', 'eclair', 'eel', 'egg_roll', 'electric_chair', 'escargot', 'eyepatch', 'falcon', 'fedora', 'ferret', 'fig_(fruit)', 'file_(tool)', 'first-aid_kit', 'fishbowl', 'flash', 'fleece', 'football_helmet', 'fudge', 'funnel', 'futon', 'gag', 'garbage', 'gargoyle', 'gasmask', 'gemstone', 'generator', 'goldfish', 'gondola_(boat)', 'gorilla', 'gourd', 'gravy_boat', 'griddle', 'grits', 'halter_top', 'hamper', 'hand_glass', 'handcuff', 'handsaw', 'hardback_book', 'harmonium', 'hatbox', 'headset', 'heron', 'hippopotamus', 'hockey_stick', 'hookah', 'hornet', 'hot-air_balloon', 'hotplate', 'hourglass', 'houseboat', 'hummus', 'popsicle', 'ice_pack', 'ice_skate', 'inhaler', 'jelly_bean', 'jewel', 'joystick', 'keg', 'kennel', 'keycard', 'kitchen_table', 'knitting_needle', 'knocker_(on_a_door)', 'koala', 'lab_coat', 'lamb-chop', 'lasagna', 'lawn_mower', 'leather', 'legume', 'lemonade', 'lightning_rod', 'limousine', 'liquor', 'machine_gun', 'mallard', 'mallet', 'mammoth', 'manatee', 'martini', 'mascot', 'masher', 'matchbox', 'microscope', 'milestone', 'milk_can', 'milkshake', 'mint_candy', 'motor_vehicle', 'music_stool', 'nailfile', 'neckerchief', 'nosebag_(for_animals)', 'nutcracker', 'octopus_(food)', 'octopus_(animal)', 'omelet', 'inkpad', 'pan_(metal_container)', 'pantyhose', 'papaya', 'paperback_book', 'paperweight', 'parchment', 'passenger_ship', 'patty_(food)', 'wooden_leg', 'pegboard', 'pencil_box', 'pencil_sharpener', 'pendulum', 'pennant', 'penny_(coin)', 'persimmon', 'phonebook', 'piggy_bank', 'pin_(non_jewelry)', 'ping-pong_ball', 'pinwheel', 'tobacco_pipe', 'pistol', 'pitchfork', 'playpen', 'plow_(farm_equipment)', 'plume', 'pocket_watch', 'poncho', 'pool_table', 'prune', 'pudding', 'puffer_(fish)', 'puffin', 'pug-dog', 'puncher', 'puppet', 'quesadilla', 'quiche', 'race_car', 'radar', 'rag_doll', 'rat', 'rib_(food)', 'river_boat', 'road_map', 'rodent', 'roller_skate', 'Rollerblade', 'root_beer', 'safety_pin', 'salad_plate', 'salmon_(food)', 'satchel', 'saucepan', 'sawhorse', 'saxophone', 'scarecrow', 'scraper', 'seaplane', 'sharpener', 'Sharpie', 'shaver_(electric)', 'shawl', 'shears', 'shepherd_dog', 'sherbert', 'shot_glass', 'shower_cap', 'shredder_(for_paper)', 'skullcap', 'sling_(bandage)', 'smoothie', 'snake', 'softball', 'sombrero', 'soup_bowl', 'soya_milk', 'space_shuttle', 'sparkler_(fireworks)', 'spear', 'crawfish', 'squid_(food)', 'stagecoach', 'steak_knife', 'stepladder', 'stew', 'stirrer', 'string_cheese', 'stylus', 'subwoofer', 'sugar_bowl', 'sugarcane_(plant)', 'syringe', 'Tabasco_sauce', 'table-tennis_table', 'tachometer', 'taco', 'tambourine', 'army_tank', 'telephoto_lens', 'tequila', 'thimble', 'trampoline', 'trench_coat', 'triangle_(musical_instrument)', 'truffle_(chocolate)', 'vat', 'turnip', 'unicycle', 'vinegar', 'violin', 'vodka', 'vulture', 'waffle_iron', 'walrus', 'wardrobe', 'washbasin', 'water_heater', 'water_gun', 'wolf'),
    BASE_CLASSES=('aerosol_can', 'air_conditioner', 'airplane', 'alarm_clock', 'alcohol', 'alligator', 'almond', 'ambulance', 'amplifier', 'anklet', 'antenna', 'apple', 'apron', 'aquarium', 'armband', 'armchair', 'artichoke', 'trash_can', 'ashtray', 'asparagus', 'atomizer', 'avocado', 'award', 'awning', 'baby_buggy', 'basketball_backboard', 'backpack', 'handbag', 'suitcase', 'bagel', 'ball', 'balloon', 'bamboo', 'banana', 'Band_Aid', 'bandage', 'bandanna', 'banner', 'barrel', 'barrette', 'barrow', 'baseball_base', 'baseball', 'baseball_bat', 'baseball_cap', 'baseball_glove', 'basket', 'basketball', 'bat_(animal)', 'bath_mat', 'bath_towel', 'bathrobe', 'bathtub', 'battery', 'bead', 'bean_curd', 'beanbag', 'beanie', 'bear', 'bed', 'bedspread', 'cow', 'beef_(food)', 'beer_bottle', 'beer_can', 'bell', 'bell_pepper', 'belt', 'belt_buckle', 'bench', 'beret', 'bib', 'bicycle', 'visor', 'billboard', 'binder', 'binoculars', 'bird', 'birdfeeder', 'birdbath', 'birdcage', 'birdhouse', 'birthday_cake', 'black_sheep', 'blackberry', 'blackboard', 'blanket', 'blazer', 'blender', 'blinker', 'blouse', 'blueberry', 'boat', 'bobbin', 'bobby_pin', 'boiled_egg', 'deadbolt', 'bolt', 'book', 'bookcase', 'booklet', 'boot', 'bottle', 'bottle_opener', 'bouquet', 'bow_(decorative_ribbons)', 'bow-tie', 'bowl', 'bowler_hat', 'box', 'suspenders', 'bracelet', 'brassiere', 'bread-bin', 'bread', 'bridal_gown', 'briefcase', 'broccoli', 'broom', 'brownie', 'brussels_sprouts', 'bucket', 'bull', 'bulldog', 'bullet_train', 'bulletin_board', 'bullhorn', 'bun', 'bunk_bed', 'buoy', 'bus_(vehicle)', 'business_card', 'butter', 'butterfly', 'button', 'cab_(taxi)', 'cabin_car', 'cabinet', 'cake', 'calculator', 'calendar', 'calf', 'camcorder', 'camel', 'camera', 'camera_lens', 'camper_(vehicle)', 'can', 'can_opener', 'candle', 'candle_holder', 'candy_cane', 'walking_cane', 'canister', 'canoe', 'cantaloup', 'cap_(headwear)', 'bottle_cap', 'cape', 'cappuccino', 'car_(automobile)', 'railcar_(part_of_a_train)', 'identity_card', 'card', 'cardigan', 'horse_carriage', 'carrot', 'tote_bag', 'cart', 'carton', 'cash_register', 'cast', 'cat', 'cauliflower', 'cayenne_(spice)', 'CD_player', 'celery', 'cellular_telephone', 'chair', 'chandelier', 'cherry', 'chicken_(animal)', 'chickpea', 'chili_(vegetable)', 'crisp_(potato_chip)', 'chocolate_bar', 'chocolate_cake', 'choker', 'chopping_board', 'chopstick', 'Christmas_tree', 'slide', 'cigarette', 'cigarette_case', 'cistern', 'clasp', 'cleansing_agent', 'clip', 'clipboard', 'clock', 'clock_tower', 'clothes_hamper', 'clothespin', 'coaster', 'coat', 'coat_hanger', 'coatrack', 'cock', 'coconut', 'coffee_maker', 'coffee_table', 'coffeepot', 'coin', 'colander', 'coleslaw', 'pacifier', 'computer_keyboard', 'condiment', 'cone', 'control', 'cookie', 'cooler_(for_food)', 'cork_(bottle_plug)', 'corkscrew', 'edible_corn', 'cornet', 'cornice', 'corset', 'costume', 'cowbell', 'cowboy_hat', 'crab_(animal)', 'cracker', 'crate', 'crayon', 'crescent_roll', 'crib', 'crock_pot', 'crossbar', 'crow', 'crown', 'crucifix', 'cruise_ship', 'police_cruiser', 'crumb', 'crutch', 'cub_(animal)', 'cube', 'cucumber', 'cufflink', 'cup', 'trophy_cup', 'cupboard', 'cupcake', 'curtain', 'cushion', 'dartboard', 'deck_chair', 'deer', 'dental_floss', 'desk', 'diaper', 'dining_table', 'dish', 'dish_antenna', 'dishrag', 'dishtowel', 'dishwasher', 'dispenser', 'Dixie_cup', 'dog', 'dog_collar', 'doll', 'dolphin', 'domestic_ass', 'doorknob', 'doormat', 'doughnut', 'drawer', 'underdrawers', 'dress', 'dress_hat', 'dress_suit', 'dresser', 'drill', 'drum_(musical_instrument)', 'duck', 'duckling', 'duct_tape', 'duffel_bag', 'dumpster', 'eagle', 'earphone', 'earring', 'easel', 'egg', 'egg_yolk', 'eggbeater', 'eggplant', 'refrigerator', 'elephant', 'elk', 'envelope', 'eraser', 'fan', 'faucet', 'Ferris_wheel', 'ferry', 'fighter_jet', 'figurine', 'file_cabinet', 'fire_alarm', 'fire_engine', 'fire_extinguisher', 'fire_hose', 'fireplace', 'fireplug', 'fish', 'fish_(food)', 'fishing_rod', 'flag', 'flagpole', 'flamingo', 'flannel', 'flap', 'flashlight', 'flip-flop_(sandal)', 'flipper_(footwear)', 'flower_arrangement', 'flute_glass', 'foal', 'folding_chair', 'food_processor', 'football_(American)', 'footstool', 'fork', 'forklift', 'freight_car', 'French_toast', 'freshener', 'frisbee', 'frog', 'fruit_juice', 'frying_pan', 'garbage_truck', 'garden_hose', 'gargle', 'garlic', 'gazelle', 'gelatin', 'giant_panda', 'gift_wrap', 'ginger', 'giraffe', 'cincture', 'glass_(drink_container)', 'globe', 'glove', 'goat', 'goggles', 'golf_club', 'golfcart', 'goose', 'grape', 'grater', 'gravestone', 'green_bean', 'green_onion', 'grill', 'grizzly', 'grocery_bag', 'guitar', 'gull', 'gun', 'hairbrush', 'hairnet', 'hairpin', 'ham', 'hamburger', 'hammer', 'hammock', 'hamster', 'hair_dryer', 'hand_towel', 'handcart', 'handkerchief', 'handle', 'hat', 'veil', 'headband', 'headboard', 'headlight', 'headscarf', 'headstall_(for_horses)', 'heart', 'heater', 'helicopter', 'helmet', 'highchair', 'hinge', 'hog', 'home_plate_(baseball)', 'honey', 'fume_hood', 'hook', 'horse', 'hose', 'hot_sauce', 'hummingbird', 'polar_bear', 'icecream', 'ice_maker', 'igniter', 'iPod', 'iron_(for_clothing)', 'ironing_board', 'jacket', 'jam', 'jar', 'jean', 'jeep', 'jersey', 'jet_plane', 'jewelry', 'jumpsuit', 'kayak', 'kettle', 'key', 'kilt', 'kimono', 'kitchen_sink', 'kite', 'kitten', 'kiwi_fruit', 'knee_pad', 'knife', 'knob', 'ladder', 'ladle', 'ladybug', 'lamb_(animal)', 'lamp', 'lamppost', 'lampshade', 'lantern', 'lanyard', 'laptop_computer', 'latch', 'legging_(clothing)', 'Lego', 'lemon', 'lettuce', 'license_plate', 'life_buoy', 'life_jacket', 'lightbulb', 'lime', 'lion', 'lip_balm', 'lizard', 'log', 'lollipop', 'speaker_(stero_equipment)', 'loveseat', 'magazine', 'magnet', 'mail_slot', 'mailbox_(at_home)', 'mandarin_orange', 'manger', 'manhole', 'map', 'marker', 'mashed_potato', 'mask', 'mast', 'mat_(gym_equipment)', 'mattress', 'measuring_cup', 'measuring_stick', 'meatball', 'medicine', 'melon', 'microphone', 'microwave_oven', 'milk', 'minivan', 'mirror', 'mitten', 'mixer_(kitchen_tool)', 'money', 'monitor_(computer_equipment) computer_monitor', 'monkey', 'motor', 'motor_scooter', 'motorcycle', 'mound_(baseball)', 'mouse_(computer_equipment)', 'mousepad', 'muffin', 'mug', 'mushroom', 'musical_instrument', 'napkin', 'necklace', 'necktie', 'needle', 'nest', 'newspaper', 'newsstand', 'nightshirt', 'noseband_(for_animals)', 'notebook', 'notepad', 'nut', 'oar', 'oil_lamp', 'olive_oil', 'onion', 'orange_(fruit)', 'orange_juice', 'ostrich', 'ottoman', 'oven', 'overalls_(clothing)', 'owl', 'packet', 'pad', 'paddle', 'padlock', 'paintbrush', 'painting', 'pajamas', 'palette', 'pan_(for_cooking)', 'pancake', 'paper_plate', 'paper_towel', 'parachute', 'parakeet', 'parasail_(sports)', 'parasol', 'parka', 'parking_meter', 'parrot', 'passenger_car_(part_of_a_train)', 'passport', 'pastry', 'pea_(food)', 'peach', 'peanut_butter', 'pear', 'peeler_(tool_for_fruit_and_vegetables)', 'pelican', 'pen', 'pencil', 'penguin', 'pepper', 'pepper_mill', 'perfume', 'person', 'pet', 'pew_(church_bench)', 'phonograph_record', 'piano', 'pickle', 'pickup_truck', 'pie', 'pigeon', 'pillow', 'pineapple', 'pinecone', 'pipe', 'pita_(bread)', 'pitcher_(vessel_for_liquid)', 'pizza', 'place_mat', 'plate', 'platter', 'pliers', 'pocketknife', 'poker_(fire_stirring_tool)', 'pole', 'polo_shirt', 'pony', 'pop_(soda)', 'postbox_(public)', 'postcard', 'poster', 'pot', 'flowerpot', 'potato', 'potholder', 'pottery', 'pouch', 'power_shovel', 'prawn', 'pretzel', 'printer', 'projectile_(weapon)', 'projector', 'propeller', 'pumpkin', 'puppy', 'quilt', 'rabbit', 'racket', 'radiator', 'radio_receiver', 'radish', 'raft', 'raincoat', 'ram_(animal)', 'raspberry', 'razorblade', 'reamer_(juicer)', 'rearview_mirror', 'receipt', 'recliner', 'record_player', 'reflector', 'remote_control', 'rhinoceros', 'rifle', 'ring', 'robe', 'rocking_chair', 'rolling_pin', 'router_(computer_equipment)', 'rubber_band', 'runner_(carpet)', 'plastic_bag', 'saddle_(on_an_animal)', 'saddle_blanket', 'saddlebag', 'sail', 'salad', 'salami', 'salmon_(fish)', 'salsa', 'saltshaker', 'sandal_(type_of_shoe)', 'sandwich', 'saucer', 'sausage', 'scale_(measuring_instrument)', 'scarf', 'school_bus', 'scissors', 'scoreboard', 'screwdriver', 'scrubbing_brush', 'sculpture', 'seabird', 'seahorse', 'seashell', 'sewing_machine', 'shaker', 'shampoo', 'shark', 'shaving_cream', 'sheep', 'shield', 'shirt', 'shoe', 'shopping_bag', 'shopping_cart', 'short_pants', 'shoulder_bag', 'shovel', 'shower_head', 'shower_curtain', 'signboard', 'silo', 'sink', 'skateboard', 'skewer', 'ski', 'ski_boot', 'ski_parka', 'ski_pole', 'skirt', 'sled', 'sleeping_bag', 'slipper_(footwear)', 'snowboard', 'snowman', 'snowmobile', 'soap', 'soccer_ball', 'sock', 'sofa', 'solar_array', 'soup', 'soupspoon', 'sour_cream', 'spatula', 'spectacles', 'spice_rack', 'spider', 'sponge', 'spoon', 'sportswear', 'spotlight', 'squirrel', 'stapler_(stapling_machine)', 'starfish', 'statue_(sculpture)', 'steak_(food)', 'steering_wheel', 'step_stool', 'stereo_(sound_system)', 'stirrup', 'stool', 'stop_sign', 'brake_light', 'stove', 'strainer', 'strap', 'straw_(for_drinking)', 'strawberry', 'street_sign', 'streetlight', 'suit_(clothing)', 'sunflower', 'sunglasses', 'sunhat', 'surfboard', 'sushi', 'mop', 'sweat_pants', 'sweatband', 'sweater', 'sweatshirt', 'sweet_potato', 'swimsuit', 'sword', 'table', 'table_lamp', 'tablecloth', 'tag', 'taillight', 'tank_(storage_vessel)', 'tank_top_(clothing)', 'tape_(sticky_cloth_or_paper)', 'tape_measure', 'tapestry', 'tarp', 'tartan', 'tassel', 'tea_bag', 'teacup', 'teakettle', 'teapot', 'teddy_bear', 'telephone', 'telephone_booth', 'telephone_pole', 'television_camera', 'television_set', 'tennis_ball', 'tennis_racket', 'thermometer', 'thermos_bottle', 'thermostat', 'thread', 'thumbtack', 'tiara', 'tiger', 'tights_(clothing)', 'timer', 'tinfoil', 'tinsel', 'tissue_paper', 'toast_(food)', 'toaster', 'toaster_oven', 'toilet', 'toilet_tissue', 'tomato', 'tongs', 'toolbox', 'toothbrush', 'toothpaste', 'toothpick', 'cover', 'tortilla', 'tow_truck', 'towel', 'towel_rack', 'toy', 'tractor_(farm_equipment)', 'traffic_light', 'dirt_bike', 'trailer_truck', 'train_(railroad_vehicle)', 'tray', 'tricycle', 'tripod', 'trousers', 'truck', 'trunk', 'turban', 'turkey_(food)', 'turtle', 'turtleneck_(clothing)', 'typewriter', 'umbrella', 'underwear', 'urinal', 'urn', 'vacuum_cleaner', 'vase', 'vending_machine', 'vent', 'vest', 'videotape', 'volleyball', 'waffle', 'wagon', 'wagon_wheel', 'walking_stick', 'wall_clock', 'wall_socket', 'wallet', 'automatic_washer', 'watch', 'water_bottle', 'water_cooler', 'water_faucet', 'water_jug', 'water_scooter', 'water_ski', 'water_tower', 'watering_can', 'watermelon', 'weathervane', 'webcam', 'wedding_cake', 'wedding_ring', 'wet_suit', 'wheel', 'wheelchair', 'whipped_cream', 'whistle', 'wig', 'wind_chime', 'windmill', 'window_box_(for_plants)', 'windshield_wiper', 'windsock', 'wine_bottle', 'wine_bucket', 'wineglass', 'blinder_(for_horses)', 'wok', 'wooden_spoon', 'wreath', 'wrench', 'wristband', 'wristlet', 'yacht', 'yogurt', 'yoke_(animal_equipment)', 'zebra', 'zucchini')
                  )


@DATASETS.register_module()
class FewShotLVISDataset_866_337(BaseFewShotDataset, LVISV1Dataset):
# class FewShotCocoDataset_65_15(BaseFewShotDataset, CocoDataset):
    """COCO dataset for any shot detection.

    Args:
        classes (str | Sequence[str] | None): Classes for model training and
            provide fixed label for each class. When classes is string,
            it will load pre-defined classes in :obj:`FewShotCocoDataset_65_15`.
            For example: 'BASE_CLASSES', 'NOVEL_CLASSES` or `ALL_CLASSES`.
        num_novel_shots (int | None): Max number of instances used for each
            novel class. If is None, all annotation will be used.
            Default: None.
        num_base_shots (int | None): Max number of instances used for each base
            class. If is None, all annotation will be used. Default: None.
        ann_shot_filter (dict | None): Used to specify the class and the
            corresponding maximum number of instances when loading
            the annotation file. For example: {'dog': 10, 'person': 5}.
            If set it as None, `ann_shot_filter` will be
            created according to `num_novel_shots` and `num_base_shots`.
        min_bbox_area (int | float | None):  Filter images with bbox whose
            area smaller `min_bbox_area`. If set to None, skip
            this filter. Default: None.
        dataset_name (str | None): Name of dataset to display. For example:
            'train dataset' or 'query dataset'. Default: None.
        test_mode (bool): If set True, annotation will not be loaded.
            Default: False.
    """

    def __init__(self,
                 classes: Optional[Union[str, Sequence[str]]] = None,
                 num_novel_shots: Optional[int] = None,
                 num_base_shots: Optional[int] = None,
                 ann_shot_filter: Optional[Dict[str, int]] = None,
                 min_bbox_area: Optional[Union[int, float]] = None,
                 dataset_name: Optional[str] = None,
                 test_mode: bool = False,
                 **kwargs) -> None:
        if dataset_name is None:
            self.dataset_name = 'Test dataset' \
                if test_mode else 'Train dataset'
        else:
            self.dataset_name = dataset_name
        self.SPLIT = LVIS_SPLIT
        assert classes is not None, f'{self.dataset_name}: classes in ' \
                                    f'`FewShotCocoDataset_65_15` can not be None.'
        # `ann_shot_filter` will be used to filter out excess annotations
        # for any shot setting. It can be configured manually or generated
        # by the `num_novel_shots` and `num_base_shots`
        self.num_novel_shots = num_novel_shots
        self.num_base_shots = num_base_shots
        self.min_bbox_area = min_bbox_area
        self.CLASSES = self.get_classes(classes) # classes is 'BASE_CLASSES'
        if ann_shot_filter is None:
            if num_novel_shots is not None or num_base_shots is not None:
                ann_shot_filter = self._create_ann_shot_filter()
        else:
            assert num_novel_shots is None and num_base_shots is None, \
                f'{self.dataset_name}: can not config ann_shot_filter and ' \
                f'num_novel_shots/num_base_shots at the same time.'

        # these values would be set in `self.load_annotations_coco`
        self.cat_ids = []
        self.cat2label = {}
        self.lvis = None
        self.img_ids = None

        super().__init__(
            classes=None,
            ann_shot_filter=ann_shot_filter,
            dataset_name=dataset_name,
            test_mode=test_mode,
            **kwargs)

    def get_classes(self, classes: Union[str, Sequence[str]]) -> List[str]:
        """Get class names.

        It supports to load pre-defined classes splits.
        The pre-defined classes splits are:
        ['ALL_CLASSES', 'NOVEL_CLASSES', 'BASE_CLASSES']

        Args:
            classes (str | Sequence[str]): Classes for model training and
                provide fixed label for each class. When classes is string,
                it will load pre-defined classes in `FewShotCocoDataset_65_15`.
                For example: 'NOVEL_CLASSES'.

        Returns:
            list[str]: list of class names.
        """
        # configure any shot classes setting
        if isinstance(classes, str):
            assert classes in self.SPLIT.keys(
            ), f'{self.dataset_name} : not a pre-defined classes or split ' \
               f'in COCO_SPLIT.'
            class_names = self.SPLIT[classes]
            if 'BASE_CLASSES' in classes:
                assert self.num_novel_shots is None, \
                    f'{self.dataset_name}: BASE_CLASSES do not have ' \
                    f'novel instances.'
            elif 'NOVEL_CLASSES' in classes:
                assert self.num_base_shots is None, \
                    f'{self.dataset_name}: NOVEL_CLASSES do not have ' \
                    f'base instances.'
        elif isinstance(classes, (tuple, list)):
            class_names = classes
        else:
            raise ValueError(f'Unsupported type {type(classes)} of classes.')
        return class_names

    def _create_ann_shot_filter(self) -> Dict:
        """Generate `ann_shot_filter` for novel and base classes.

        Returns:
            dict[str, int]: The number of shots to keep for each class.
        """
        ann_shot_filter = {}
        # generate annotation filter for novel classes
        if self.num_novel_shots is not None:
            for class_name in self.SPLIT['NOVEL_CLASSES']:
                ann_shot_filter[class_name] = self.num_novel_shots
        # generate annotation filter for base classes
        if self.num_base_shots is not None:
            for class_name in self.SPLIT['BASE_CLASSES']:
                ann_shot_filter[class_name] = self.num_base_shots
        return ann_shot_filter

    def load_annotations(self, ann_cfg: List[Dict]) -> List[Dict]:
        """Support to Load annotation from two type of ann_cfg.

            - type of 'ann_file': COCO-style annotation file.
            - type of 'saved_dataset': Saved COCO dataset json.

        Args:
            ann_cfg (list[dict]): Config of annotations.

        Returns:
            list[dict]: Annotation infos.
        """
        data_infos = []
        for ann_cfg_ in ann_cfg:
            if ann_cfg_['type'] == 'saved_dataset':
                data_infos += self.load_annotations_saved(ann_cfg_['ann_file'])
            elif ann_cfg_['type'] == 'ann_file':
                data_infos += self.load_annotations_lvis(ann_cfg_['ann_file'])
            else:
                raise ValueError(f'not support annotation type '
                                 f'{ann_cfg_["type"]} in ann_cfg.')
        return data_infos

    def load_annotations_lvis(self, ann_file: str) -> List[Dict]:
        """Load annotation from COCO style annotation file.

        Args:
            ann_file (str): Path of annotation file.

        Returns:
            list[dict]: Annotation info from COCO api.
        """
        self.lvis = LVIS(ann_file)
        self.coco_format_ann = COCO(ann_file)
        # to keep the label order equal to the order in CLASSES
        if len(self.cat_ids) == 0:
            for i, class_name in enumerate(self.CLASSES):
                cat_id = self.coco_format_ann.get_cat_ids(cat_names=[class_name])[0]
                self.cat_ids.append(cat_id)
                self.cat2label[cat_id] = i
        else:
            # check categories id consistency between different files
            for i, class_name in enumerate(self.CLASSES):
                cat_id = self.coco_format_ann.get_cat_ids(cat_names=[class_name])[0]
                assert self.cat2label[cat_id] == i, \
                    'please make sure all the json files use same ' \
                    'categories id for same class'
        # len(self.cat2label) is 866
        # len(self.cat_ids) is 866
        self.img_ids = self.lvis.get_img_ids()
        
        data_infos = []
        total_ann_ids = []
        for i in self.img_ids:
            info = self.lvis.load_imgs([i])[0]
            info['filename'] = info['coco_url'].split('/')[-1]
            info['ann'] = self._get_ann_info(info)
            # to support different version of coco since some annotation file
            # contain images from train2014 and val2014 at the same time
            if 'train2014' in info['filename']:
                info['filename'] = 'train2014/' + info['filename']
            elif 'val2014' in info['filename']:
                info['filename'] = 'val2014/' + info['filename']
            elif 'instances_val2017' in ann_file:
                info['filename'] = 'val2017/' + info['filename']
            elif 'instances_train2017' in ann_file:
                info['filename'] = 'train2017/' + info['filename']
            data_infos.append(info)
            ann_ids = self.lvis.get_ann_ids(img_ids=[i])
            total_ann_ids.extend(ann_ids)
        assert len(set(total_ann_ids)) == len(
            total_ann_ids
        ), f'{self.dataset_name}: Annotation ids in {ann_file} are not unique!'
        return data_infos

    def _get_ann_info(self, data_info: Dict) -> Dict:
        """Get COCO annotation by index.

        Args:
            data_info(dict): Data info.

        Returns:
            dict: Annotation info of specified index.
        """

        img_id = data_info['id']
        ann_ids = self.lvis.get_ann_ids(img_ids=[img_id])
        ann_info = self.lvis.load_anns(ann_ids)
        return self._parse_ann_info(data_info, ann_info)

    def get_cat_ids(self, idx: int) -> List[int]:
        """Get category ids by index.

        Overwrite the function in CocoDataset.

        Args:
            idx (int): Index of data.

        Returns:
            list[int]: All categories in the image of specified index.
        """

        return self.data_infos[idx]['ann']['labels'].astype(np.int64).tolist()

    def _filter_imgs(self,
                     min_size: int = 32,
                     min_bbox_area: Optional[int] = None) -> List[int]:
        """Filter images that do not meet the requirements.

        Args:
            min_size (int): Filter images with length or width
                smaller than `min_size`. Default: 32.
            min_bbox_area (int | None): Filter images with bbox whose
                area smaller `min_bbox_area`. If set to None, skip
                this filter. Default: None.

        Returns:
            list[int]: valid indices of data_infos.
        """
        valid_inds = []
        valid_img_ids = []
        if min_bbox_area is None:
            min_bbox_area = self.min_bbox_area
        for i, img_info in enumerate(self.data_infos):
            # filter empty image
            if self.filter_empty_gt and img_info['ann']['labels'].size == 0:
                continue
            # filter images smaller than `min_size`
            if min(img_info['width'], img_info['height']) < min_size:
                continue
            # filter image with bbox smaller than min_bbox_area
            # it is usually used in Attention RPN
            if min_bbox_area is not None:
                skip_flag = False
                for bbox in img_info['ann']['bboxes']:
                    bbox_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                    if bbox_area < min_bbox_area:
                        skip_flag = True
                if skip_flag:
                    continue
            valid_inds.append(i)
            valid_img_ids.append(img_info['id'])
        # update coco img_ids
        self.img_ids = valid_img_ids
        return valid_inds

    def evaluate(self,
                 results: List[Sequence],
                 metric: Union[str, List[str]] = 'bbox',
                 logger: Optional[object] = None,
                 jsonfile_prefix: Optional[str] = None,
                 classwise: bool = False,
                 proposal_nums: Sequence[int] = (100, 300, 1000),
                 iou_thrs: Optional[Union[float, Sequence[float]]] = None,
                 metric_items: Optional[Union[List[str], str]] = None,
                 class_splits: Optional[List[str]] = None) -> Dict:
        """Evaluation in COCO protocol and summary results of different splits
        of classes.

        Args:
            results (list[list | tuple]): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated. Options are
                'bbox', 'proposal', 'proposal_fast'. Default: 'bbox'
            logger (logging.Logger | None): Logger used for printing
                related information during evaluation. Default: None.
            jsonfile_prefix (str | None): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.
            classwise (bool): Whether to evaluating the AP for each class.
            proposal_nums (Sequence[int]): Proposal number used for evaluating
                recalls, such as recall@100, recall@1000.
                Default: (100, 300, 1000).
            iou_thrs (Sequence[float] | float | None): IoU threshold used for
                evaluating recalls/mAPs. If set to a list, the average of all
                IoUs will also be computed. If not specified, [0.50, 0.55,
                0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95] will be used.
                Default: None.
            metric_items (list[str] | str | None): Metric items that will
                be returned. If not specified, ``['AR@100', 'AR@300',
                'AR@1000', 'AR_s@1000', 'AR_m@1000', 'AR_l@1000' ]`` will be
                used when ``metric=='proposal'``, ``['mAP', 'mAP_50', 'mAP_75',
                'mAP_s', 'mAP_m', 'mAP_l']`` will be used when
                ``metric=='bbox'``.
            class_splits: (list[str] | None): Calculate metric of classes split
                in COCO_SPLIT. For example: ['BASE_CLASSES', 'NOVEL_CLASSES'].
                Default: None.

        Returns:
            dict[str, float]: COCO style evaluation metric.
        """
        if class_splits is not None:
            for k in class_splits:
                assert k in self.SPLIT.keys(), 'please define classes split.'
        metrics = metric if isinstance(metric, list) else [metric]
        allowed_metrics = ['bbox', 'proposal', 'proposal_fast']
        for metric in metrics:
            if metric not in allowed_metrics:
                raise KeyError(f'metric {metric} is not supported')
        if iou_thrs is None:
            iou_thrs = np.linspace(
                .5, 0.95, int(np.round((0.95 - .5) / .05)) + 1, endpoint=True)
        if metric_items is not None:
            if not isinstance(metric_items, list):
                metric_items = [metric_items]

        result_files, tmp_dir = self.format_results(results, jsonfile_prefix)

        eval_results = OrderedDict()
        # cocoGt = self.lvis
        cocoGt = self.coco_format_ann
        for metric in metrics:
            msg = f'Evaluating {metric}...'
            if logger is None:
                msg = '\n' + msg
            print_log(msg, logger=logger)

            iou_type = 'bbox' if metric == 'proposal' else metric
            if metric not in result_files:
                raise KeyError(f'{metric} is not in results')
            try:
                predictions = mmcv.load(result_files[metric])
                cocoDt = cocoGt.loadRes(predictions)
            except IndexError:
                print_log(
                    'The testing results of the whole dataset is empty.',
                    logger=logger,
                    level=logging.ERROR)
                break

            if metric is 'bbox':
                lvis_eval = LVISEval(cocoGt, cocoDt, 'bbox')
                lvis_eval.run()
                lvis_eval.print_results()

            # eval each class splits
            if class_splits is not None:
                class_splits = {k: LVIS_SPLIT[k] for k in class_splits}
                for split_name in class_splits.keys():
                    split_cat_ids = [
                        self.cat_ids[i] for i in range(len(self.CLASSES))
                        if self.CLASSES[i] in class_splits[split_name]
                    ]
                    self._evaluate_by_class_split(
                        cocoGt,
                        cocoDt,
                        iou_type,
                        proposal_nums,
                        iou_thrs,
                        split_cat_ids,
                        metric,
                        metric_items,
                        eval_results,
                        False,
                        logger,
                        split_name=split_name + ' ')
            # eval all classes
            self._evaluate_by_class_split(cocoGt, cocoDt, iou_type,
                                          proposal_nums, iou_thrs,
                                          self.cat_ids, metric, metric_items,
                                          eval_results, classwise, logger)

        if tmp_dir is not None:
            tmp_dir.cleanup()
        return eval_results

    def _evaluate_by_class_split(self,
                                 cocoGt: object,
                                 cocoDt: object,
                                 iou_type: str,
                                 proposal_nums: Sequence[int],
                                 iou_thrs: Union[float, Sequence[float]],
                                 cat_ids: List[int],
                                 metric: str,
                                 metric_items: Union[str, List[str]],
                                 eval_results: Dict,
                                 classwise: bool,
                                 logger: object,
                                 split_name: str = '') -> Dict:
        """Evaluation a split of classes in COCO protocol.

        Args:
            cocoGt (object): coco object with ground truth annotations.
            cocoDt (object): coco object with detection results.
            iou_type (str): Type of IOU.
            proposal_nums (Sequence[int]): Number of proposals.
            iou_thrs (float | Sequence[float]): Thresholds of IoU.
            cat_ids (list[int]): Class ids of classes to be evaluated.
            metric (str): Metrics to be evaluated.
            metric_items (str | list[str]): Metric items that will
                be returned. If not specified, ``['AR@100', 'AR@300',
                'AR@1000', 'AR_s@1000', 'AR_m@1000', 'AR_l@1000' ]`` will be
                used when ``metric=='proposal'``, ``['mAP', 'mAP_50', 'mAP_75',
                'mAP_s', 'mAP_m', 'mAP_l']`` will be used when
                ``metric=='bbox'``.
            eval_results (dict[str, float]): COCO style evaluation metric.
            classwise (bool): Whether to evaluating the AP for each class.
            split_name (str): Name of split. Default:''.

        Returns:
            dict[str, float]: COCO style evaluation metric.
        """
        

        cocoEval = COCOeval(cocoGt, cocoDt, iou_type)
        cocoEval.params.imgIds = self.img_ids
        cocoEval.params.maxDets = list(proposal_nums)
        cocoEval.params.iouThrs = iou_thrs

        cocoEval.params.catIds = cat_ids
        # mapping of cocoEval.stats
        coco_metric_names = {
            'mAP': 0,
            'mAP_50': 1,
            'mAP_75': 2,
            'mAP_s': 3,
            'mAP_m': 4,
            'mAP_l': 5,
            'AR@100': 6,
            'AR@300': 7,
            'AR@1000': 8,
            'AR_s@1000': 9,
            'AR_m@1000': 10,
            'AR_l@1000': 11,
            'AR@100_50': 12,
        }
        if metric_items is not None:
            for metric_item in metric_items:
                if metric_item not in coco_metric_names:
                    raise KeyError(
                        f'metric item {metric_item} is not supported')
        if split_name is not None:
            print_log(f'\n evaluation of {split_name} class', logger=logger)
        if metric == 'proposal':
            cocoEval.params.useCats = 0
            cocoEval.evaluate()
            cocoEval.accumulate()
            cocoEval.summarize()
            if metric_items is None:
                metric_items = [
                    'AR@100', 'AR@300', 'AR@1000', 'AR_s@1000', 'AR_m@1000',
                    'AR_l@1000'
                ]

            for item in metric_items:
                val = float(f'{cocoEval.stats[coco_metric_names[item]]:.3f}')
                eval_results[split_name + item] = val
        else:
            cocoEval.evaluate()
            cocoEval.accumulate()
            cocoEval.summarize()
            if classwise:  # Compute per-category AP
                # Compute per-category AP
                # from https://github.com/facebookresearch/detectron2/
                precisions = cocoEval.eval['precision']
                # precision: (iou, recall, cls, area range, max dets)
                assert len(self.cat_ids) == precisions.shape[2], \
                    f'{self.cat_ids},{precisions.shape}'

                results_per_category = []
                for idx, catId in enumerate(self.cat_ids):
                    # area range index 0: all area ranges
                    # max dets index -1: typically 100 per image
                    nm = self.lvis.loadCats(catId)[0]
                    precision = precisions[:, :, idx, 0, -1]
                    precision = precision[precision > -1]
                    if precision.size:
                        ap = np.mean(precision)
                    else:
                        ap = float('nan')
                    results_per_category.append(
                        (f'{nm["name"]}', f'{float(ap):0.3f}'))

                num_columns = min(6, len(results_per_category) * 2)
                results_flatten = list(itertools.chain(*results_per_category))
                headers = [split_name + 'category', split_name + 'AP'] * (
                    num_columns // 2)
                results_2d = itertools.zip_longest(*[
                    results_flatten[i::num_columns] for i in range(num_columns)
                ])
                table_data = [headers]
                table_data += [result for result in results_2d]
                table = AsciiTable(table_data)
                print_log('\n' + table.table, logger=logger)

            if metric_items is None:
                metric_items = [
                    'mAP', 'mAP_50', 'mAP_75', 'mAP_s', 'mAP_m', 'mAP_l'
                ]

            for metric_item in metric_items:
                key = f'{metric}_{metric_item}'
                val = float(
                    f'{cocoEval.stats[coco_metric_names[metric_item]]:.3f}')
                eval_results[split_name + key] = val
            ap = cocoEval.stats[:6]
            eval_results[split_name + f'{metric}_mAP_copypaste'] = (
                f'{ap[0]:.3f} {ap[1]:.3f} {ap[2]:.3f} {ap[3]:.3f} '
                f'{ap[4]:.3f} {ap[5]:.3f}')

            return eval_results




