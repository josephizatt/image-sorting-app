package main

import (
	"encoding/json"
	"fmt"
	"io"
	"io/ioutil"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"sort"

	tf "github.com/tensorflow/tensorflow/tensorflow/go"
)

func main() {
	http.HandleFunc("/upload", uploadImage)
	log.Fatal(http.ListenAndServe(":8080", nil))
}

func uploadImage(w http.ResponseWriter, r *http.Request) {
	err := r.ParseMultipartForm(10 << 20) // Limit the size of the uploaded file to 10MB
	if err != nil {
		http.Error(w, "Unable to process the uploaded image", http.StatusBadRequest)
		return
	}

	file, handler, err := r.FormFile("image")
	if err != nil {
		http.Error(w, "Error retrieving the image file", http.StatusBadRequest)
		return
	}
	defer file.Close()

	filePath := filepath.Join("uploads", handler.Filename)
	outFile, err := os.Create(filePath)
	if err != nil {
		http.Error(w, "Error saving the uploaded image", http.StatusInternalServerError)
		return
	}
	defer outFile.Close()

	_, err = io.Copy(outFile, file)
	if err != nil {
		http.Error(w, "Error saving the uploaded image", http.StatusInternalServerError)
		return
	}

	model, err := tf.LoadSavedModel("efficientnet", []string{"serve"}, nil)
	if err != nil {
		http.Error(w, "Failed to load the EfficientNet model", http.StatusInternalServerError)
		return
	}

	imgTensor, err := loadImageForPrediction(filePath)
	if err != nil {
		http.Error(w, "Error loading the image for prediction", http.StatusInternalServerError)
		return
	}

	outputs, err := model.Session.Run(
		map[tf.Output]*tf.Tensor{
			model.Graph.Operation("input_1").Output(0): imgTensor,
		},
		[]tf.Output{
			model.Graph.Operation("probs").Output(0),
		},
		nil,
	)

	if err != nil {
		http.Error(w, "Error performing the image prediction", http.StatusInternalServerError)
		return
	}

	probs := outputs[0].Value().([][]float32)[0]
	predictions := make([]Prediction, len(probs))
	for i, prob := range probs {
		predictions[i] = Prediction{ClassID: i, Probability: prob}
	}

	// Sort predictions by probability in descending order
	sort.Slice(predictions, func(i, j int) bool {
		return predictions[i].Probability > predictions[j].Probability
	})

	// Get the top prediction
	topPrediction := predictions[0]

	// TODO: Assign appropriate tags to the image based on the predicted class

	response := struct {
		Message     string  `json:"message"`
		Prediction  string  `json:"prediction"`
		Probability float32 `json:"probability"`
	}{
		Message:     "Image uploaded and processed successfully",
		Prediction:  fmt.Sprintf("Class %d", topPrediction.ClassID),
		Probability: topPrediction.Probability,
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

type Prediction struct {
	ClassID     int
	Probability float32
}

func loadImageForPrediction(filePath string) (*tf.Tensor, error) {
	imgData, err := ioutil.ReadFile(filePath)
	if err != nil {
		return nil, err
	}

	imgTensor, err := tf.NewTensor(string(imgData))
	if err != nil {
		return nil, err
	}

	return imgTensor, nil
}
