### Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/ghiyatssyahied/onlineExamSurveillance-V2.git
   cd onlineExamSurveillance-V2
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Download the **Shape Predictor 68** model (Cant push because too big file):
   ```bash
   wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
   bzip2 -d shape_predictor_68_face_landmarks.dat.bz2
   ```
## Usage
### Running the System
To start real-time monitoring, run:
```bash
python main.py
```
