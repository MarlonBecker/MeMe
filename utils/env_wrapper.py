import numpy as np

try:
    from simFrame.environment import Environment
except ImportError:
    #create dummy Env class
    print("ERROR: could not import simulation backend")
    cont = input("Continue with dummy environment? (Y/n): ")
    if not cont in ["y", "Y", ""]:
        exit()
    class Environment():
        def __init__(self, **config):
            self.designArea = config["designArea"]
            self.structure = np.ones(self.designArea)
        def setFOM(self, *args, **kwargs):
            pass
        def setStructure(self, *args, **kwargs):
            pass
        def evaluate(self, *args, **kwargs):
            return 1, None, None, None
        def flipPixel(self, *args, **kwargs):
            pass

from utils.FOM_factory import FigureOfMeritFactory

class SimEnv():
    def __init__(self, config_dict, reset_structure: np.ndarray = None, runInitSim: bool = True):
        super().__init__()
        self.config = config_dict
        self.env = None # env init deferred to self.reset()
        self.structure = None

        self.reset_structure = reset_structure if reset_structure is not None else np.ones(self.config["environment"]["designArea"])

        self.last_fields = None
        self.fields = None
        self.efficiency = None
        self.reset_efficiency = None
        self.reset_fields = None

        self.action_type = self.config["action_type"]
        self.default_solver = self.config["environment"].get("default_solver", "jaxwell-bicgstab")

        if runInitSim:
            self.reset()

    def reset(self, reset_structure: np.ndarray = None):
        if reset_structure is None:
            reset_structure = self.reset_structure

        self.structure = reset_structure
        self.fields = None

        # first call, so set up self.env
        if self.env == None:
            self.env = Environment(**self.config["environment"])
            self.env.setFOM(getattr(FigureOfMeritFactory, self.config["figureOfMerit"]))
            self.setStructure(reset_structure)
            self.reset_efficiency, self.reset_fields, overlaps, powers = self.evaluate()
            print(f"Reset Results: Efficiency: {self.reset_efficiency:.3f}. Overlaps: {overlaps}")
        else:
            self.setStructure(reset_structure)

        self.efficiency = self.reset_efficiency
        self.fields = self.reset_fields

        return self.structure

    def step(self, action: int):
        #remote envs lead to wrapping in dict
        if isinstance(action, dict):
            action = action["agent0"]

        self.last_fields = self.fields

        # obtain list of pixel indices (2D!) to flip
        pixels_to_flip = self.get_flip_indices(action)

        # flip pixels based on action type based pixel indices
        for pixel in pixels_to_flip:
            self.env.flipPixel(pixel)
        self.structure = self.env.structure.copy()

        #fixed areas:
        if "fixed_pixels" in self.config["environment"]:
            self.structure = self.env.structure.copy()
            for i in self.config["environment"]["fixed_pixels"]:
                self.structure[i["area"][0][0]:i["area"][1][0],i["area"][0][1]:i["area"][1][1]] = i["value"]
            self.env.setStructure(self.structure)

        self.efficiency, self.fields, overlaps, powers = self.evaluate()

        return self.efficiency, overlaps, powers

    def revert(self, action: int):
        # obtain list of pixel indices (2D!) to flip
        pixels_to_flip = self.get_flip_indices(action)

        # flip pixels based on action type based pixel indices
        for pixel in pixels_to_flip:
            self.env.flipPixel(pixel)
        self.structure = self.env.structure.copy()

        self.fields = self.last_fields


    def evaluate(self):
        return self.env.evaluate(method="fdfd", E_initial=self.fields, solver=self.default_solver)[:4]


    def get_action_space_shape(self):
        if self.action_type == "each_pixel_once":
            return self.env.designArea
        elif self.action_type == "symmetric_x":
            return self.env.designArea[0]//2, self.env.designArea[1]
        elif self.action_type == "symmetric_y":
            return self.env.designArea[0], self.env.designArea[1]//2
        elif self.action_type == "symmetric_x_y":
            return self.env.designArea[0]//2, self.env.designArea[1]//2

    def get_action_space_size(self):
        shape = self.get_action_space_shape()
        return shape[0]*shape[1]

    def get_optimizable_structure(self):
        if self.action_type == "each_pixel_once":
            return self.structure.copy()
        elif self.action_type == "symmetric_x":
            return self.structure.copy()[:self.env.designArea[0]//2,:]
        elif self.action_type == "symmetric_y":
            return self.structure.copy()[:,:self.env.designArea[1]//2]
        elif self.action_type == "symmetric_x_y":
            return self.structure.copy()[:self.env.designArea[0]//2,:self.env.designArea[1]//2]

    def get_flip_indices(self, action):
        n_pixels: int = self.env.designArea[0] * self.env.designArea[1]
        pixels_to_flip = None
        if self.action_type == "each_pixel_once":
            pixels_to_flip = [divmod(action, self.env.designArea[1])]
        elif self.action_type == "symmetric_x":
            x_top = int(divmod(action, self.env.designArea[1])[0])
            y_top = int(divmod(action, self.env.designArea[1])[1])
            x_bottom = int(self.env.designArea[0]-1-x_top)
            y_bottom = y_top
            # just mirror the given action on the other side of the axis
            pixels_to_flip = [
                [x_top, y_top],
                [x_bottom, y_bottom]
            ]
        elif self.action_type == "symmetric_y":
            x_left = int(divmod(action, self.env.designArea[1]/2)[0])
            y_left = int(divmod(action, self.env.designArea[1]/2)[1])
            x_right = x_left
            y_right = int(self.env.designArea[1]-1-y_left)
            # just mirror the given action on the other side of the axis
            pixels_to_flip = [
                [x_left, y_left],
                [x_right, y_right]
            ]
        elif self.action_type == "symmetric_x_y":
            x_left_top = int(divmod(action, self.env.designArea[1]/2)[0])
            y_left_top = int(divmod(action, self.env.designArea[1]/2)[1])
            x_right_top = int(self.env.designArea[0]-1-x_left_top)
            y_right_top = y_left_top
            x_left_bottom = x_left_top
            y_left_bottom =  int(self.env.designArea[1]-1-y_left_top)
            x_right_bottom = int(self.env.designArea[0]-1-x_left_top)
            y_right_bottom = y_left_bottom
            # just mirror the given action on the other side of the axis
            pixels_to_flip = [
                [x_left_top, y_left_top],
                [x_right_top, y_right_top],
                [x_left_bottom, y_left_bottom],
                [x_right_bottom, y_right_bottom]
            ]

        return pixels_to_flip

    def setStructure(self, structure):
        return self.env.setStructure(structure)
