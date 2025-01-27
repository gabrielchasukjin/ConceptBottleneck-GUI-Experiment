import React, { useState, useEffect } from "react";
import axios from "axios";
import { Box, Button, Typography, TextField } from "@mui/material";
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend,
} from "chart.js";
import { Bar } from "react-chartjs-2";
import "./App.css";

ChartJS.register(CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend);

function App() {
  const [isTraining, setIsTraining] = useState(false);
  const [inputText, setInputText] = useState("");
  const [classificationResult, setClassificationResult] = useState("");
  const [activations, setActivations] = useState({});
  const [selectedDataset, setSelectedDataset] = useState("");
  const [selectedModel, setSelectedModel] = useState("");
  const [hardwareInfo, setHardwareInfo] = useState(null);

  const handleTrainClick = async () => {
    setIsTraining(true);
    try {
      const response = await axios.post("http://localhost:5050/train", {
        dataset: selectedDataset,
        model: selectedModel
      });
      alert("Training completed: " + response.data.message);
    } catch (error) {
      console.error("Train Error:", error.response?.data || error.message);
      alert("Training failed. Check the backend logs.");
    } finally {
      setIsTraining(false);
    }
  };

  const handleClassifyClick = async () => {
    try {
      const response = await axios.post("http://localhost:5050/test", {
        input: inputText,
      });
      setClassificationResult(`${response.data.classification} (${response.data.confidence}% confidence)`);

      const activationsObj = response.data.top_concepts.reduce((acc, concept) => {
        acc[concept.concept] = concept.activation;
        return acc;
      }, {});
      setActivations(activationsObj);
    } catch (error) {
      console.error("Classify Error:", error.response?.data || error.message);
      alert("Classification failed. Check the backend logs.");
    }
  };

  const isTrainingConfigured = () => {
    return selectedDataset && selectedModel;
  };

  const getDatasetDescription = () => {
    const descriptions = {
      'SST2': "SST2 (Socher et al., 2013): comprise 6920 training samples, 872 validation samples, and 1821 test samples of movie reviews with positive and negative classes.",
      'yelp_polarity': "Yelp Polarity (YelpP) (Zhang et al., 2015): comprise 560,000 training samples and 38,000 test samples of Yelp reviews with positive and negative classes.",
      'ag_news': "AGnews (Zhang et al., 2015): comprise 120,000 training samples and 7,600 test samples of news articles with 4 classes.",
      'dbpedia_14': "DBpedia (Lehmann et al., 2015): comprise 560,000 training samples and 70,000 test samples from DBpedia 2014 with 14 classes."
    };
    return descriptions[selectedDataset] || "Select a dataset to see its description";
  };

  const fetchHardwareInfo = async () => {
    try {
      const response = await axios.get("http://localhost:5050/hardware-info");
      setHardwareInfo(response.data);
    } catch (error) {
      console.error("Error fetching hardware info:", error);
    }
  };

  useEffect(() => {
    fetchHardwareInfo();
  }, []);

  return (
    <Box className="app-container">
      <Box className="header-container">
        <Typography variant="h6" className="app-title">
          CB-GUI
        </Typography>
      </Box>

      {hardwareInfo && (
        <Box className="hardware-info-container">
          <Box className="hardware-info">
            <Typography variant="caption" className="hardware-text">
              <span className="hardware-label">CPU:</span>{" "}
              <span className="hardware-value">{hardwareInfo.cpu}</span>
            </Typography>
            <Typography variant="caption" className="hardware-text">
              <span className="hardware-label">RAM:</span>{" "}
              <span className="hardware-value">{hardwareInfo.ram}</span>
            </Typography>
            <Typography variant="caption" className="hardware-text">
              <span className="hardware-label">GPU:</span>{" "}
              <span className="hardware-value">{hardwareInfo.gpu || 'None'}</span>
            </Typography>
          </Box>
        </Box>
      )}

      <Box className="main-content">
        <Box className="section-box">
          <Typography variant="h6" className="section-title">
            Train CB Layer
          </Typography>
          
          <Typography variant="subtitle1" className="option-label">
            Select Dataset:
          </Typography>
          <Box className="options-container">
            {["SST2", "yelp_polarity", "ag_news", "dbpedia_14"].map((dataset) => (
              <Button
                key={dataset}
                variant={selectedDataset === dataset ? "contained" : "outlined"}
                onClick={() => setSelectedDataset(dataset)}
                className="option-button"
              >
                {dataset}
              </Button>
            ))}
          </Box>

          <Typography variant="subtitle1" className="option-label">
            Select Model:
          </Typography>
          <Box className="options-container">
            {["gpt2", "roberta"].map((model) => (
              <Button
                key={model}
                variant={selectedModel === model ? "contained" : "outlined"}
                onClick={() => setSelectedModel(model)}
                className="option-button"
              >
                {model}
              </Button>
            ))}
          </Box>

          <Button
            variant="contained"
            color="primary"
            onClick={handleTrainClick}
            disabled={!isTrainingConfigured() || isTraining}
            className="train-button"
          >
            {isTraining ? "Training..." : "Train"}
          </Button>
        </Box>
        
        <Box className="right-column">
          <Box className="info-box">
            <Typography variant="h6" className="section-title">
              Selected Options Info
            </Typography>
            {selectedDataset && (
              <Box className="info-section">
                <Typography variant="subtitle2" className="info-label">
                  Dataset Description:
                </Typography>
                <Typography variant="body2" className="info-text">
                  {getDatasetDescription()}
                </Typography>
              </Box>
            )}
          </Box>

          <Box className="section-box">
            <Typography variant="h6" className="section-title">
              Classify Input
            </Typography>
            <TextField
              label="Input Text"
              value={inputText}
              onChange={(e) => setInputText(e.target.value)}
              fullWidth
              margin="normal"
            />
            <Button
              variant="contained"
              color="secondary"
              onClick={handleClassifyClick}
              disabled={!inputText}
              className="classify-button"
            >
              Classify
            </Button>
            {classificationResult && (
              <Typography marginTop="10px">
                Classification: {classificationResult}
              </Typography>
            )}
          </Box>
        </Box>
      </Box>

      {Object.keys(activations).length > 0 && (
        <Box className="chart-container">
          <Typography variant="h6" marginBottom="10px">
            Classification Contributions
          </Typography>
          <Box className="chart-box">
            <Bar
              data={{
                labels: Object.keys(activations),
                datasets: [
                  {
                    label: "Activation Levels",
                    data: Object.values(activations),
                    backgroundColor: "rgba(0, 123, 255, 0.6)",
                    borderColor: "rgba(0, 123, 255, 1)",
                    borderWidth: 1,
                  },
                ],
              }}
              options={{
                responsive: true,
                maintainAspectRatio: false,
                indexAxis: 'y',
                plugins: {
                  legend: { 
                    display: true,
                    labels: {
                      font: {
                        family: "'Roboto', 'Helvetica', 'Arial', sans-serif",
                        size: 12
                      }
                    }
                  },
                },
                scales: {
                  x: { 
                    beginAtZero: true,
                    title: {
                      display: true,
                      text: 'Activation Values',
                      font: {
                        size: 14,
                        weight: 500
                      }
                    },
                    grid: {
                      color: 'rgba(0, 0, 0, 0.05)'
                    }
                  },
                  y: { 
                    beginAtZero: true,
                    title: {
                      display: true,
                      text: 'Concepts',
                      font: {
                        size: 14,
                        weight: 500
                      }
                    },
                    grid: {
                      color: 'rgba(0, 0, 0, 0.05)'
                    }
                  },
                },
              }}
            />
          </Box>
        </Box>
      )}
    </Box>
  );
}

export default App;
