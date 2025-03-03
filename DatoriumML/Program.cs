using DatoriumML;
using Microsoft.ML;
using Microsoft.ML.Data;

string _assetsPath = Path.Combine(Environment.CurrentDirectory, "Prerequisites/assets");
string _imagesFolder = Path.Combine(_assetsPath, "images");
string _trainTagsTsv = Path.Combine(_imagesFolder, "tags.tsv");
string _testTagsTsv = Path.Combine(_imagesFolder, "test-tags.tsv");
string _predictSingleImage = Path.Combine(_imagesFolder, "toaster3.jpg");
string _inceptionTensorFlowModel = Path.Combine(_assetsPath, "inception", "tensorflow_inception_graph.pb");

MLContext mlContext = new MLContext();

void DisplayResult(IEnumerable<ImagePrediction> predictions)
{
    foreach (var prediction in predictions)
    {
        Console.WriteLine($"Image: {Path.GetFileName(prediction.ImagePath)} predicted as: {prediction.PredictedLabelValue} with score: {prediction.Score?.Max()} ");
    }
}