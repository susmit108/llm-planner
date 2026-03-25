class Rules:
    def __init__(self, persona:dict, task:dict):
        self._persona = persona
        self.task = task
        self.process()

    def generate_task(self, ref_task:str):
        convo_len = len(self.task)
        self.task[f'{convo_len+1}'] = ref_task
    def sleep_blood_pressure(self):
        sleep_hr = self._persona['Average_Sleeping_Hours']
        bp = self._persona['Has_High_Blood_Pressure']
        if sleep_hr!=-1 and bp!=-1:
            if sleep_hr<8 and bp==1:
                self.generate_task("Try to get more sleep to reduce BP")
    
    def exercise_weight(self):
        exercise = self._persona['Daily_Exercise']
        weight = self._persona['Weight']
        if exercise!=-1 and weight!=-1:
            if exercise!=1 and weight>90:
                self.generate_task("Try to do more exercise to reduce weight")

    def process(self):
        self.sleep_blood_pressure()
        self.exercise_weight()
        return self.task



    
