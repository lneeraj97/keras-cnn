# keras-cnn
### A CNN for classification of retinal fundus images built using Keras with tensorflow backend
## Project Structure
<body>
    <h1>Directory Tree</h1>
    <p>
        <a>./</a><br>
        ├── <a>data</a><br>
        │   ├── <a>test</a><br>
        │   │   ├── <a>0</a><br>
        │   │   ├── <a>1</a><br>
        │   │   ├── <a>2</a><br>
        │   │   └── <a>3</a><br>
        │   └── <a>train</a><br>
        │   &nbsp;&nbsp;&nbsp; ├── <a>0</a><br>
        │   &nbsp;&nbsp;&nbsp; ├── <a>1</a><br>
        │   &nbsp;&nbsp;&nbsp; ├── <a>2</a><br>
        │   &nbsp;&nbsp;&nbsp; └── <a>3</a><br>
        ├── <a></a>environment.yml</a><br>
        ├── <a>model</a><br>
        │   ├── <a>model.h5</a><br>
        │   ├── <a>model.json</a><br>
        │   ├── <a>model.png</a><br>
        │   └── <a>model.txt</a><br>
        ├── <a>result.html</a><br>
        └── <a>src</a><br>
        &nbsp;&nbsp;&nbsp; ├── <a>export.py</a><br>
        &nbsp;&nbsp;&nbsp; ├── <a>model</a><br>
        &nbsp;&nbsp;&nbsp; │   ├── <a>cnn.py</a><br>
        &nbsp;&nbsp;&nbsp; │   ├── <a>__init__.py</a><br>
        &nbsp;&nbsp;&nbsp; │   └── <a>__pycache__</a><br>
        &nbsp;&nbsp;&nbsp; ├── <a>test.py</a><br>
        &nbsp;&nbsp;&nbsp; └── <a>train.py</a><br>
        <br><br>
    </p>
    <p>

  
</body>


## To run the project,do:
1. Install miniconda or anaconda
2. Open the project folder in terminal
3. Load the environment.yml file using the command below
> conda env create -f environment.yml 
4. Activate the environment in conda
5. Run the src/train.py to train the model
> python src/train.py
6. Export the neural network model to text,image and json using the src/export.py file
> python src/export.py
7. Test the neural network using external images with src/test.py
>python src/test.py
