import React, { useState } from "react";
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

ChartJS.register(CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend);

function App() {
  const [isTraining, setIsTraining] = useState(false);
  const [inputText, setInputText] = useState("");
  const [classificationResult, setClassificationResult] = useState("");
  const [activations, setActivations] = useState({});
  const [hardwareOption, setHardwareOption] = useState("Local Hardware");

  const handleTrainClick = async () => {
    setIsTraining(true);
    try {
      const response = await axios.post("http://localhost:5050/train", {
        hardware: hardwareOption,
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
      setClassificationResult(response.data.classification);

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

  const handleHardwareClick = (option) => {
    setHardwareOption(option);
  };

  return (
    <Box display="flex" flexDirection="column" alignItems="center" padding="20px">
      <Box
        display="flex"
        justifyContent="flex-start"
        alignItems="center"
        width="100%"
        marginBottom="20px"
      >
        <Typography variant="h6" style={{ fontWeight: "bold", fontSize: "14px" }}>
          CB-GUI
        </Typography>
      </Box>

      <Box
        display="flex"
        justifyContent="space-between"
        alignItems="flex-start"
        width="80%"
        marginBottom="40px"
      >
        <Box
          display="flex"
          flexDirection="column"
          alignItems="center"
          width="30%"
          padding="20px"
          border="1px solid lightgray"
          borderRadius="8px"
          height="170px"
          margin="20px"
        >
          <Typography variant="h6" marginBottom="10px" style={{ textAlign: "center" }}>
            Select Hardware
          </Typography>
          <Button
            variant={hardwareOption === "Local Hardware" ? "contained" : "outlined"}
            onClick={() => handleHardwareClick("Local Hardware")}
            style={{ marginBottom: "10px" }}
          >
            Local Hardware
          </Button>
          <Button
            variant={hardwareOption === "DSMLP" ? "contained" : "outlined"}
            onClick={() => handleHardwareClick("DSMLP")}
          >
            DSMLP
          </Button>
        </Box>

        <Box
          display="flex"
          flexDirection="column"
          alignItems="center"
          width="30%"
          padding="20px"
          border="1px solid lightgray"
          borderRadius="8px"
          height="170px"
          margin="20px"
        >
          <Typography variant="h6" marginBottom="10px" style={{ textAlign: "center" }}>
            Train CB Layer
          </Typography>
          <Button
            variant="contained"
            color="primary"
            onClick={handleTrainClick}
            disabled={isTraining}
          >
            {isTraining ? "Training..." : "Train"}
          </Button>
        </Box>

        <Box
          display="flex"
          flexDirection="column"
          alignItems="center"
          width="30%"
          padding="20px"
          border="1px solid lightgray"
          borderRadius="8px"
          height="170px"
          margin="20px"
        >
          <Typography variant="h6" marginBottom="10px" style={{ textAlign: "center" }}>
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

      {Object.keys(activations).length > 0 && (
        <Box
          display="flex"
          flexDirection="column"
          justifyContent="center"
          alignItems="center"
          width="60%"
          padding="20px"
          border="1px solid lightgray"
          borderRadius="8px"
          marginTop="20px"
        >
          <Typography variant="h6" marginBottom="10px">
            Classification Contributions
          </Typography>
          <Box width="100%" height="400px">
            <Bar
              data={{
                labels: Object.keys(activations),
                datasets: [
                  {
                    label: "Activation Levels",
                    data: Object.values(activations),
                    backgroundColor: "rgba(75, 192, 192, 0.6)",
                    borderColor: "rgba(75, 192, 192, 1)",
                    borderWidth: 1,
                  },
                ],
              }}
              options={{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                  legend: { display: true },
                },
                scales: {
                  x: { beginAtZero: true },
                  y: { beginAtZero: true },
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
