


class Face:
    def __init__(self, id: int, frame, left: int, top: int, right: int, bottom: int):
        self.__id = id
        self.__frame = frame
        self.__left = int(left)
        self.__top = int(top)
        self.__right = int(right)
        self.__bottom = int(bottom)
        self.__age = None
        self.__gender = None
        self.__distance = None
        self.__mesh_frame = None
        self.__mesh_points = None
        self.__is_mask = False
        self.__is_distance = False
        self.__is_age = False
        self.__is_gender = False
        self.__is_mesh_frame = False
        self.__is_mesh_points = False
        self.__is_descriptor = False

    def get_id(self) -> int:
        return self.__id

    def get_frame(self) -> None:
        return self.__frame

    def get_left(self) -> int:
        return self.__left

    def get_top(self) -> int:
        return self.__top

    def get_right(self) -> int:
        return self.__right

    def get_bottom(self) -> int:
        return self.__bottom

    def get_coordinates(self) -> (int, int, int, int):
        return self.__left, self.__top, self.__right, self.__bottom

    def set_is_mask(self, is_mask: bool):
        self.__is_mask = is_mask

    def get_is_mask(self) -> bool:
        return self.__is_mask

    def set_distance(self, distance: float) -> None:
        self.__distance = distance
        self.__is_distance = True

    def get_distance(self) -> float:
        if not self.__is_distance:
            raise Exception('Distance is not exist yet!')
        return self.__distance

    def set_age(self, age: float) -> None:
        self.__age = age
        self.__is_age = True

    def get_age(self) -> float:
        if not self.__is_age:
            raise Exception('Age is not exist yet!')
        return self.__age

    def set_gender(self, gender: bool) -> None:
        self.__gender = gender
        self.__is_gender = True

    def get_gender(self) -> bool:
        if not self.__is_gender:
            raise Exception('Gender is not exist yet!')
        return self.__gender

    def set_mesh_frame(self, frame) -> None:
        self.__mesh_frame = frame
        self.__is_mesh_frame = True

    def get_mesh_frame(self):
        if not self.__is_mesh_frame:
            raise Exception('Mesh frame are not exist yet!')
        return self.__mesh_frame

    def set_mesh_points(self, mesh_points) -> None:
        self.__mesh_points = mesh_points
        self.__is_mesh_points = True

    def get_mesh_points(self):
        if not self.__is_mesh_points:
            raise Exception('Mesh points are not exist yet!')
        return self.__mesh_points
