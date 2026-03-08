from enum import Enum
import os

class SensorChannel(str, Enum):
    """Enumeration for the 9 IMU sensor columns."""
    UACC_X = 'uacc_x'
    UACC_Y = 'uacc_y'
    UACC_Z = 'uacc_z'
    GYR_X = 'gyr_x'
    GYR_Y = 'gyr_y'
    GYR_Z = 'gyr_z'
    GRAV_X = 'grav_x'
    GRAV_Y = 'grav_y'
    GRAV_Z = 'grav_z'

class FileNames(str, Enum):
    MODEL_NAME = 'elite_ensemble.pkl'
    LABELS_NAME = 'elite_labels.pkl'
    SCALER_NAME = 'elite_scaler.pkl'

class MovementClass(str, Enum):
    """Enumeration for the 43 movement categories."""
    BALL = 'Ball'
    BEAR = 'Bear'
    BOOK = 'Book'
    BUILDING_BLOCKS = 'Building_blocks'
    CLAPPING = 'Clapping'
    CRAB = 'Crab'
    CUPBOARD = 'Cupboard'
    DOORHANDLE = 'Doorhandle'
    DRAWING = 'Drawing'
    DRINKING = 'Drinking'
    DWARF = 'Dwarf'
    FACE_WASH = 'Face_wash'
    FROG = 'Frog'
    GLASS_GRABBING = 'Glass_grabbing'
    GLASS_LIFTING = 'Glass_lifting'
    GLASS_TO_MOUTH = 'Glass_to_mouth'
    GOLIATH = 'Goliath'
    HAND_WASH = 'Hand_wash'
    HOPSCOTCH = 'Hopscotch'
    KNEE_OTHER = 'Knee_other'
    KNEE_SAME = 'Knee_same'
    LAME_FOX = 'Lame_fox'
    LIGHT_OFF = 'Light_off'
    LIGHT_ON = 'Light_on'
    NOSE = 'Nose'
    PECK = 'Peck'
    PRAY = 'Pray'
    PUDING_EAT = 'Puding_eat'
    PUDING_OPEN = 'Puding_open'
    RABBIT = 'Rabbit'
    SEAL = 'Seal'
    SHOE_OFF_OTHER = 'Shoe_off_other'
    SHOE_OFF_SAME = 'Shoe_off_same'
    SHOE_ON_OTHER = 'Shoe_on_other'
    SHOE_ON_SAME = 'Shoe_on_same'
    SNACK_EAT = 'Snack_eat'
    SNACK_OPEN = 'Snack_open'
    SOCK_OFF_OTHER = 'Sock_off_other'
    SOCK_OFF_SAME = 'Sock_off_same'
    SOCK_ON_OTHER = 'Sock_on_other'
    SOCK_ON_SAME = 'Sock_on_same'
    SPIDER = 'Spider'
    SWIMMING = 'Swimming'
    TOOTHBRUSH_OTHER = 'Toothbrush_other'
    TOOTHBRUSH_SAME = 'Toothbrush_same'
    # Superclass
    FOOTWEAR = 'Footwear'
    GLASS_HANDLING = 'Glass_Handling'
    TABLE_PLAY = 'Table_Play'
    LEG_TOUCH = 'Leg_Touch'
    LIGHTING = 'Lighting'
    EATING = 'Eating'
    CRWLING_PLAY = 'Crawling_Play'

BLACKLIST = [
    MovementClass.PUDING_OPEN,
    MovementClass.SNACK_OPEN,
    MovementClass.FROG,
    MovementClass.LAME_FOX,
    MovementClass.TOOTHBRUSH_SAME,
    MovementClass.TOOTHBRUSH_OTHER,
    MovementClass.CUPBOARD,
]

MAPPING = {
    MovementClass.SHOE_ON_SAME: MovementClass.FOOTWEAR,
    MovementClass.SHOE_OFF_SAME: MovementClass.FOOTWEAR,
    MovementClass.SOCK_ON_SAME: MovementClass.FOOTWEAR,
    MovementClass.SOCK_OFF_SAME: MovementClass.FOOTWEAR,
    MovementClass.SHOE_ON_OTHER: MovementClass.FOOTWEAR,
    MovementClass.SHOE_OFF_OTHER: MovementClass.FOOTWEAR,
    MovementClass.SOCK_ON_OTHER: MovementClass.FOOTWEAR,
    MovementClass.SOCK_OFF_OTHER: MovementClass.FOOTWEAR,

    MovementClass.PUDING_EAT: MovementClass.EATING,
    MovementClass.SNACK_EAT: MovementClass.EATING,

    MovementClass.BEAR: MovementClass.CRWLING_PLAY,
    MovementClass.RABBIT: MovementClass.CRWLING_PLAY,
    MovementClass.SEAL: MovementClass.CRWLING_PLAY,
    MovementClass.SPIDER: MovementClass.CRWLING_PLAY,
    MovementClass.CRAB: MovementClass.CRWLING_PLAY,

    MovementClass.BOOK: MovementClass.TABLE_PLAY,
    MovementClass.BUILDING_BLOCKS: MovementClass.TABLE_PLAY,
    MovementClass.PECK: MovementClass.TABLE_PLAY,

    MovementClass.GLASS_GRABBING: MovementClass.GLASS_HANDLING,
    MovementClass.GLASS_LIFTING: MovementClass.GLASS_HANDLING,
    MovementClass.GLASS_TO_MOUTH: MovementClass.GLASS_HANDLING,

    MovementClass.LIGHT_ON: MovementClass.LIGHTING,
    MovementClass.LIGHT_OFF: MovementClass.LIGHTING,

    MovementClass.KNEE_SAME: MovementClass.LEG_TOUCH,
    MovementClass.KNEE_OTHER: MovementClass.LEG_TOUCH,
}

class Config:
    """Central configuration for paths and hyperparameters."""
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    ROOT_DIR = os.path.dirname(CURRENT_DIR)

    output_path = '/after_early_stopping'

    RAW_DATA_DIR = os.path.join(CURRENT_DIR, 'movements')
    VISUALS_DIR = os.path.join(CURRENT_DIR, 'visuals'+output_path)
    MODELS_DIR = os.path.join(ROOT_DIR, 'models'+output_path)

    SAMPLING_RATE_HZ = 50
    WINDOW_SEC = 2
    OVERLAP_PCT = 0.5

    BATCH_SIZE = 32
    LEARNING_RATE = .001
    EPOCHS = 40

    WEIGHT_DECAY = 1e-4

    SENSOR_FEATURES = [
        SensorChannel.UACC_X.value, SensorChannel.UACC_Y.value, SensorChannel.UACC_Z.value,
        SensorChannel.GYR_X.value, SensorChannel.GYR_Y.value, SensorChannel.GYR_Z.value,
        SensorChannel.GRAV_X.value, SensorChannel.GRAV_Y.value, SensorChannel.GRAV_Z.value
    ]