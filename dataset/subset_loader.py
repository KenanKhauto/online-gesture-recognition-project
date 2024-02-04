from typing import List, Union, Optional

from .loader import GestureDataset

class GestureSubset(GestureDataset):
    def __init__(
            self,
            hdf5_path,
            label_file,
            transform=None,
            sample_duration=142,
            class_ids: Optional[List[int]]=None,
            n_samples_per_class: Union[int, List[int], None]=None,
        ):
        self.class_ids = class_ids
        if isinstance(n_samples_per_class, int) and class_ids:
            self.n_samples_per_class = {class_id: n_samples_per_class for class_id in class_ids}
        elif n_samples_per_class:
            self.n_samples_per_class = {class_id: n for class_id, n in zip(class_ids, n_samples_per_class)}
        else:
            self.n_samples_per_class = None
        super().__init__(
            hdf5_path,
            label_file,
            transform,
            sample_duration,
        )
    
    def parse_labels(self, label_file):
        """
        Parse the label file and return a structure containing
        the mappings of video name, start frame, end frame, label, label id and the number of frames.
        
        Parameters:
            label_file: .txt file that contains the labels
        
        Returns:
            list of tuples: (folder_name, start_frame, end_frame, label, label id, number of frames)
        """
        labels = []
        samples_per_class = {}
        with open(label_file, 'r') as file:
            for line in file:
                folder_name, label, id, start, end, number_frames = line.split(",")
                class_id = int(id) - 1

                if self.class_ids and class_id not in self.class_ids:
                    continue
                if self.n_samples_per_class and samples_per_class.get(class_id, 0) >= self._get_max_samples_for_class(class_id):
                    continue

                if self.n_samples_per_class and class_id not in samples_per_class:
                    samples_per_class[class_id] = 1
                elif self.n_samples_per_class:
                    samples_per_class[class_id] += 1

                labels.append((folder_name, int(start), int(end), label, class_id, int(number_frames)))
        return labels
    
    def _get_max_samples_for_class(self, class_id):
        if isinstance(self.n_samples_per_class, int):
            return self.n_samples_per_class
        else:
            return self.n_samples_per_class[class_id]