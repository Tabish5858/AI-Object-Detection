OBJECT_DESCRIPTIONS = {
    "person": "A human being, capable of walking upright and using complex tools. Humans are characterized by their intelligence and ability to create and use technology.",
    "car": "A four-wheeled motor vehicle designed for transportation. Modern cars typically have internal combustion engines or electric motors.",
    "bus": "A large motor vehicle designed to carry multiple passengers, commonly used for public transportation in cities and between cities.",
    "dog": "A domesticated mammal of the family Canidae, known as 'man's best friend'. Dogs are kept as pets and working animals.",
    "cat": "A small domesticated carnivorous mammal, popular as a house pet. Known for their independent nature and hunting abilities.",
    "bicycle": "A human-powered vehicle with two wheels, pedals, and handlebars. Used for transportation, exercise, and recreation.",
    "bottle": "A container designed to store and transport liquids, typically made of glass or plastic with a narrow neck.",
    "chair": "A piece of furniture designed for sitting, typically having a back and four legs.",
    "sofa": "A long upholstered seat with a back and arms, designed to seat multiple people comfortably.",
    "train": "A connected series of railroad cars moved by a locomotive, used for transporting passengers or freight.",
    "aeroplane": "A powered flying vehicle with fixed wings and a weight greater than that of the air it displaces.",
    "boat": "A watercraft designed for transportation on water, ranging from small personal vessels to large ships.",
    "motorbike": "A two-wheeled motor vehicle, also known as a motorcycle, used for personal transportation.",
    "tvmonitor": "An electronic display device used to show video and images, commonly used with computers or for watching television.",
    "horse": "A large hoofed mammal, historically used for transportation and now often kept for recreation and sports.",
    "sheep": "A domesticated ruminant mammal kept for its wool, meat, and milk.",
    "cow": "A domesticated bovine mammal kept for milk and meat production.",
    "diningtable": "A piece of furniture with a flat top and legs, used for dining and serving meals.",
    "pottedplant": "A plant grown in a container, typically used for indoor or patio decoration.",
    "bird": "A warm-blooded egg-laying vertebrate distinguished by feathers, wings, and a beak."
}

def get_description(class_name, color="unknown"):
    """Get detailed description for a class name with optional color information"""
    try:
        basic_desc = OBJECT_DESCRIPTIONS.get(class_name.lower(), "No description available")
        if color and color != "unknown":
            return f"A {color} colored {class_name.lower()}. {basic_desc}"
        return basic_desc
    except Exception as e:
        return f"An object of type {class_name}. No detailed description available."