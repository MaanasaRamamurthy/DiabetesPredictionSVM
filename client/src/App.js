import React , {useState, useEffect} from 'react'
import Axios from 'axios'
import "./css/App.css"
function App() {
  const [data,setData] = useState([{}])
  const [output , setOutput] = useState('')

  const loadData = async() =>{
    const result = await Axios.get("http://localhost:5000");          
        setData(result.data)      
}
useEffect(() => {
  loadData();
},[]);
console.log(data)
const [formData, setFormData] = React.useState(
  {
    pregnancies:'',
    Glucose:'',
    BP:'',
    Skin:'',
    Insulin:'',
    BMI:'',
    Pedigree:'',
    Age:''
}
)

function handleSubmit(e){
  e.preventDefault()
  Axios({
    method: 'POST',
    url: 'http://localhost:5000',
    data: {
      pregnancies:formData.pregnancies,
      Glucose:formData.Glucose,
      BP:formData.BP,
      Skin:formData.Skin,
      Insulin:formData.Insulin,
      BMI:formData.BMI,
      Pedigree:formData.Pedigree,
      Age:formData.Age
    }
  })
  .then((response) => {
    setOutput(response.data)
    console.log(output)
    console.log(response)
  })
  .catch((error) => {
    console.log(error)
  })
}
function handleChange(event){

setFormData(values => ({...values, [event.target.name]: event.target.value}))
}
console.log(formData)
  return (
    <div className='main_body_content'>
    <h1> Diabetes Prediction System</h1>
    <p>Predicts if a person is diabetic or non-diabetic</p>
    <form 
    onSubmit={(e)=>{handleSubmit(e);}}
    >
      <div className='input_field'>
      <div >
        <div className="group">
          <input name="pregnancies" type='number' onChange={handleChange} value={formData.pregnancies}/>
          <span className="highlight"></span>
          <span className="bar"></span>
          <label>Number of pregnancies</label>
        </div>
        <div className="group">
        
        <input name="Glucose" type="number" onChange={handleChange} value={formData.Glucose}/>
        <span className="bar"></span>
        <label>Blood Glucose level</label>
        </div>
      
        <div className="group">
        
        <input name="BP" type="number" onChange={handleChange} value={formData.BP}/>
        <label>Blood Pressure</label>
        <span className="bar"></span>
        </div>
      
        <div className="group">
        <input name="Skin" type="number" onChange={handleChange} value={formData.Skin}/>
        <label>Skin Thickness</label>
        <span className="bar"></span>
        </div>
      </div>
      <div>
        <div className="group">
        <input name="Insulin" type="number" onChange={handleChange} value={formData.Insulin}/>
        <label>Blood insulin level</label>

        <span className="bar"></span>
        </div>
      
        <div className="group">
        <input name="BMI" type="number" onChange={handleChange} value={formData.BMI}/>
        <label>Body mass index</label>

        <span className="bar"></span>
        </div>
      
        <div className="group">
        <input name="Pedigree" type="number" onChange={handleChange} value={formData.Pedigree}/>
        <label>Diabetes Pedigree Function</label>

        <span className="bar"></span>
        </div>
      
        <div className="group">
        <input name="Age" type="number" onChange={handleChange} value={formData.Age}/>
        <label>Age</label>

        <span className="bar"></span>
        </div>
      
      </div>
      </div>
      
      <div>
      <button 
      className="btn btn-submit"
      onClick={(e)=>{handleSubmit(e);}}
      >Submit</button>
      </div>
      <p>{output}</p>
    </form>

  </div>

  );
}

export default App;
