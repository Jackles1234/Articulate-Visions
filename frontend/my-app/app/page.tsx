'use client'
import { useState } from 'react';
import axios from 'axios';

const IndexPage = () => {
  const [formData, setFormData] = useState({
    textbox1: '',
    textbox2: '',
    textbox3: '',
  });

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value } = e.target;
    setFormData(prevState => ({
      ...prevState,
      [name]: value,
    }));
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    try {
      const dataToSend = JSON.stringify(formData);
      await axios.post('http://localhost:5000/api/submit', dataToSend, {
        headers: {
          'Content-Type': 'application/json',
        },
      });
      alert('Data submitted successfully!');
    } catch (error) {
      console.error('Error submitting data:', error);
      alert('Error submitting data');
    }
  };

  return (
    <div>
      <form onSubmit={handleSubmit}>
        <input
        //batch_size
          type="text"
          name="textbox1"
          value={formData.textbox1}
          onChange={handleChange}
        />
        <input
        //n_epoch
          type="text"
          name="textbox2"
          value={formData.textbox2}
          onChange={handleChange}
        />
        <input
        //timesteps
          type="text"
          name="textbox3"
          value={formData.textbox3}
          onChange={handleChange}
        />
        <button type="submit">Submit</button>
      </form>
    </div>
  );
};

export default IndexPage;