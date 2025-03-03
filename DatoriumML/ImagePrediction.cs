using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DatoriumML
{
    public class ImagePrediction : ImageData
    {
        public float[]? Score;

        public string? PredictedLabelValue;
    }
}
