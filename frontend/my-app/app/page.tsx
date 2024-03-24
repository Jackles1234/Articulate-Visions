'use client'
import React, { useState, ChangeEvent, FormEvent, useEffect } from 'react';
import axios from 'axios';
import './globals.css';

interface FormData {
  textbox1: string;
  textbox2: string;
  textbox3: string;
}

interface ImageData {
  name: string;
  url: string;
}

const IndexPage: React.FC = () => {
  const [formData, setFormData] = useState<FormData>({
    textbox1: '',
    textbox2: '',
    textbox3: '',
  });

  const [images, setImages] = useState<ImageData[]>([]);

  const [sidebarOpen, setSidebarOpen] = useState<boolean>(true);

  useEffect(() => {
    fetchImages();
  }, []);

  const fetchImages = async () => {
    try {
      const response = await axios.get<ImageData[]>('http://localhost:5000/api/images');
      setImages(response.data);
    } catch (error) {
      console.error('Error fetching images:', error);
    }
  };

  const handleChange = (e: ChangeEvent<HTMLInputElement>) => {
    const { name, value } = e.target;
    setFormData(prevState => ({
      ...prevState,
      [name]: value,
    }));
  };

  const handleSubmit = async (e: FormEvent<HTMLFormElement>) => {
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
      <nav className="navbar">
        <div>
          <span className="font-semibold text-xl tracking-tight">Your Logo</span>
        </div>
        <div>
          <a href="#" className="navbar-link">Home</a>
          <a href="#" className="navbar-link">About</a>
        </div>
      </nav>

      <button className="sidebar-button" onClick={() => setSidebarOpen(!sidebarOpen)}>Toggle Sidebar</button>
      <div className={`sidebar ${sidebarOpen ? 'open' : ''}`}>
        <div>
          <h2 className="text-lg font-semibold">Sidebar</h2>
          <form onSubmit={handleSubmit} className="sidebar-form">
            <input
              type="text"
              name="textbox1"
              value={formData.textbox1}
              onChange={handleChange}
              placeholder="Textbox 1"
              className="sidebar-input"
            />
            <input
              type="text"
              name="textbox2"
              value={formData.textbox2}
              onChange={handleChange}
              placeholder="Textbox 2"
              className="sidebar-input"
            />
            <input
              type="text"
              name="textbox3"
              value={formData.textbox3}
              onChange={handleChange}
              placeholder="Textbox 3"
              className="sidebar-input"
            />
            <button type="submit" className="sidebar-button-submit">Submit</button>
          </form>
        </div>
      </div>

      <div className="images">
        <h2 className="text-lg font-semibold">Images</h2>
        <div className="image-grid">
          {images.map((image, index) => (
            <div key={index} className="image-container">
              <img src={`http://localhost:5000${image.url}`} alt={image.name} />
              <p className="image-info">{image.name}</p>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

export default IndexPage;