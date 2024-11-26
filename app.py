import os
import warnings

import cv2
import joblib
import numpy as np
import pandas as pd
import requests
from flask import Flask
from flask import jsonify
from flask import request
from flask_cors import CORS
from PIL import Image
from skimage.feature import graycomatrix
from skimage.feature import graycoprops
from skimage.feature import local_binary_pattern

# Configurar Flask
app = Flask(__name__)
CORS(app)  # Permitir conexiones desde otros dominios
app.config["UPLOAD_FOLDER"] = "uploads"  # Carpeta para las imágenes cargadas
app.config["ALLOWED_EXTENSIONS"] = {"png", "jpg", "jpeg"}  # Extensiones permitidas


@app.before_first_request
def load_model():
    try:
        app.config["MODEL"] = download_model_from_drive(
            "1A2B3C4D5E6F7G8H9"
        )  # Reemplaza con tu ID
    except Exception as e:
        print(f"Error al cargar el modelo: {e}")


def download_model_from_drive(drive_id, output_path="papas.pkl"):
    url = f"https://drive.google.com/uc?id={drive_id}"
    response = requests.get(url, stream=True)

    if response.status_code == 200:
        with open(output_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024):
                f.write(chunk)
        print(f"Modelo descargado y guardado en {output_path}")
        return joblib.load(output_path)
    else:
        raise Exception(f"Error al descargar el modelo: {response.status_code}")


# Función para verificar si el archivo tiene una extensión permitida
def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in app.config["ALLOWED_EXTENSIONS"]
    )


# Función para extraer características de color
def extract_color_features(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    hist = cv2.calcHist(
        [hsv_image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256]
    )
    hist = cv2.normalize(hist, hist).flatten()
    return hist


# Función para extraer características de textura
def extract_texture_features(gray_image):
    lbp = local_binary_pattern(gray_image, P=8, R=1, method="uniform")
    glcm = graycomatrix(
        gray_image, distances=[5], angles=[0], levels=256, symmetric=True, normed=True
    )
    contrast = graycoprops(glcm, "contrast")[0, 0]
    return [float(lbp.mean()), float(contrast)]  # Convertir a float


# Función para extraer características de forma
def extract_shape_features(gray_image):
    contours, _ = cv2.findContours(
        gray_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    max_contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(max_contour)
    perimeter = cv2.arcLength(max_contour, True)
    circularity = 4 * np.pi * (area / (perimeter * perimeter)) if perimeter != 0 else 0
    return [area, perimeter, circularity]


# Función para procesar una sola imagen
def process_single_image(image_path):
    try:
        # Leer la imagen
        image = np.array(Image.open(image_path))
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Extraer características
        color_features = extract_color_features(image)
        texture_features = extract_texture_features(gray_image)
        shape_features = extract_shape_features(gray_image)

        # Combinar todas las características en una lista
        features = color_features.tolist() + texture_features + shape_features
        return features

    except Exception as e:
        print(f"Error al procesar la imagen {image_path}: {e}")
        return None


# Función para cargar el modelo y realizar la predicción
def predict_image_class(image_path, model_path="papas.pkl"):
    # Suprimir advertencias sobre los nombres de características
    warnings.filterwarnings("ignore", message=".*does not have valid feature names.*")

    # Cargar el modelo .pkl
    try:
        model = joblib.load(model_path)
    except Exception as e:
        print(f"Error al cargar el modelo: {e}")
        return None

    # Procesar la imagen y extraer las características
    features = process_single_image(image_path)

    if features is not None:
        # Convertir las características a un DataFrame con los nombres de las características (si el modelo fue entrenado con nombres)
        feature_names = [
            "Color_0",
            "Color_1",
            "Color_2",
            "Color_3",
            "Color_4",
            "Color_5",
            "Color_6",
            "Color_7",
            "Color_8",
            "Color_9",
            "Color_10",
            "Color_11",
            "Color_12",
            "Color_13",
            "Color_14",
            "Color_15",
            "Color_16",
            "Color_17",
            "Color_18",
            "Color_19",
            "Color_20",
            "Color_21",
            "Color_22",
            "Color_23",
            "Color_24",
            "Color_25",
            "Color_26",
            "Color_27",
            "Color_28",
            "Color_29",
            "Color_30",
            "Color_31",
            "Color_32",
            "Color_33",
            "Color_34",
            "Color_35",
            "Color_36",
            "Color_37",
            "Color_38",
            "Color_39",
            "Color_40",
            "Color_41",
            "Color_42",
            "Color_43",
            "Color_44",
            "Color_45",
            "Color_46",
            "Color_47",
            "Color_48",
            "Color_49",
            "Color_50",
            "Color_51",
            "Color_52",
            "Color_53",
            "Color_54",
            "Color_55",
            "Color_56",
            "Color_57",
            "Color_58",
            "Color_59",
            "Color_60",
            "Color_61",
            "Color_62",
            "Color_63",
            "Color_64",
            "Color_65",
            "Color_66",
            "Color_67",
            "Color_68",
            "Color_69",
            "Color_70",
            "Color_71",
            "Color_72",
            "Color_73",
            "Color_74",
            "Color_75",
            "Color_76",
            "Color_77",
            "Color_78",
            "Color_79",
            "Color_80",
            "Color_81",
            "Color_82",
            "Color_83",
            "Color_84",
            "Color_85",
            "Color_86",
            "Color_87",
            "Color_88",
            "Color_89",
            "Color_90",
            "Color_91",
            "Color_92",
            "Color_93",
            "Color_94",
            "Color_95",
            "Color_96",
            "Color_97",
            "Color_98",
            "Color_99",
            "Color_100",
            "Color_101",
            "Color_102",
            "Color_103",
            "Color_104",
            "Color_105",
            "Color_106",
            "Color_107",
            "Color_108",
            "Color_109",
            "Color_110",
            "Color_111",
            "Color_112",
            "Color_113",
            "Color_114",
            "Color_115",
            "Color_116",
            "Color_117",
            "Color_118",
            "Color_119",
            "Color_120",
            "Color_121",
            "Color_122",
            "Color_123",
            "Color_124",
            "Color_125",
            "Color_126",
            "Color_127",
            "Color_128",
            "Color_129",
            "Color_130",
            "Color_131",
            "Color_132",
            "Color_133",
            "Color_134",
            "Color_135",
            "Color_136",
            "Color_137",
            "Color_138",
            "Color_139",
            "Color_140",
            "Color_141",
            "Color_142",
            "Color_143",
            "Color_144",
            "Color_145",
            "Color_146",
            "Color_147",
            "Color_148",
            "Color_149",
            "Color_150",
            "Color_151",
            "Color_152",
            "Color_153",
            "Color_154",
            "Color_155",
            "Color_156",
            "Color_157",
            "Color_158",
            "Color_159",
            "Color_160",
            "Color_161",
            "Color_162",
            "Color_163",
            "Color_164",
            "Color_165",
            "Color_166",
            "Color_167",
            "Color_168",
            "Color_169",
            "Color_170",
            "Color_171",
            "Color_172",
            "Color_173",
            "Color_174",
            "Color_175",
            "Color_176",
            "Color_177",
            "Color_178",
            "Color_179",
            "Color_180",
            "Color_181",
            "Color_182",
            "Color_183",
            "Color_184",
            "Color_185",
            "Color_186",
            "Color_187",
            "Color_188",
            "Color_189",
            "Color_190",
            "Color_191",
            "Color_192",
            "Color_193",
            "Color_194",
            "Color_195",
            "Color_196",
            "Color_197",
            "Color_198",
            "Color_199",
            "Color_200",
            "Color_201",
            "Color_202",
            "Color_203",
            "Color_204",
            "Color_205",
            "Color_206",
            "Color_207",
            "Color_208",
            "Color_209",
            "Color_210",
            "Color_211",
            "Color_212",
            "Color_213",
            "Color_214",
            "Color_215",
            "Color_216",
            "Color_217",
            "Color_218",
            "Color_219",
            "Color_220",
            "Color_221",
            "Color_222",
            "Color_223",
            "Color_224",
            "Color_225",
            "Color_226",
            "Color_227",
            "Color_228",
            "Color_229",
            "Color_230",
            "Color_231",
            "Color_232",
            "Color_233",
            "Color_234",
            "Color_235",
            "Color_236",
            "Color_237",
            "Color_238",
            "Color_239",
            "Color_240",
            "Color_241",
            "Color_242",
            "Color_243",
            "Color_244",
            "Color_245",
            "Color_246",
            "Color_247",
            "Color_248",
            "Color_249",
            "Color_250",
            "Color_251",
            "Color_252",
            "Color_253",
            "Color_254",
            "Color_255",
            "Color_256",
            "Color_257",
            "Color_258",
            "Color_259",
            "Color_260",
            "Color_261",
            "Color_262",
            "Color_263",
            "Color_264",
            "Color_265",
            "Color_266",
            "Color_267",
            "Color_268",
            "Color_269",
            "Color_270",
            "Color_271",
            "Color_272",
            "Color_273",
            "Color_274",
            "Color_275",
            "Color_276",
            "Color_277",
            "Color_278",
            "Color_279",
            "Color_280",
            "Color_281",
            "Color_282",
            "Color_283",
            "Color_284",
            "Color_285",
            "Color_286",
            "Color_287",
            "Color_288",
            "Color_289",
            "Color_290",
            "Color_291",
            "Color_292",
            "Color_293",
            "Color_294",
            "Color_295",
            "Color_296",
            "Color_297",
            "Color_298",
            "Color_299",
            "Color_300",
            "Color_301",
            "Color_302",
            "Color_303",
            "Color_304",
            "Color_305",
            "Color_306",
            "Color_307",
            "Color_308",
            "Color_309",
            "Color_310",
            "Color_311",
            "Color_312",
            "Color_313",
            "Color_314",
            "Color_315",
            "Color_316",
            "Color_317",
            "Color_318",
            "Color_319",
            "Color_320",
            "Color_321",
            "Color_322",
            "Color_323",
            "Color_324",
            "Color_325",
            "Color_326",
            "Color_327",
            "Color_328",
            "Color_329",
            "Color_330",
            "Color_331",
            "Color_332",
            "Color_333",
            "Color_334",
            "Color_335",
            "Color_336",
            "Color_337",
            "Color_338",
            "Color_339",
            "Color_340",
            "Color_341",
            "Color_342",
            "Color_343",
            "Color_344",
            "Color_345",
            "Color_346",
            "Color_347",
            "Color_348",
            "Color_349",
            "Color_350",
            "Color_351",
            "Color_352",
            "Color_353",
            "Color_354",
            "Color_355",
            "Color_356",
            "Color_357",
            "Color_358",
            "Color_359",
            "Color_360",
            "Color_361",
            "Color_362",
            "Color_363",
            "Color_364",
            "Color_365",
            "Color_366",
            "Color_367",
            "Color_368",
            "Color_369",
            "Color_370",
            "Color_371",
            "Color_372",
            "Color_373",
            "Color_374",
            "Color_375",
            "Color_376",
            "Color_377",
            "Color_378",
            "Color_379",
            "Color_380",
            "Color_381",
            "Color_382",
            "Color_383",
            "Color_384",
            "Color_385",
            "Color_386",
            "Color_387",
            "Color_388",
            "Color_389",
            "Color_390",
            "Color_391",
            "Color_392",
            "Color_393",
            "Color_394",
            "Color_395",
            "Color_396",
            "Color_397",
            "Color_398",
            "Color_399",
            "Color_400",
            "Color_401",
            "Color_402",
            "Color_403",
            "Color_404",
            "Color_405",
            "Color_406",
            "Color_407",
            "Color_408",
            "Color_409",
            "Color_410",
            "Color_411",
            "Color_412",
            "Color_413",
            "Color_414",
            "Color_415",
            "Color_416",
            "Color_417",
            "Color_418",
            "Color_419",
            "Color_420",
            "Color_421",
            "Color_422",
            "Color_423",
            "Color_424",
            "Color_425",
            "Color_426",
            "Color_427",
            "Color_428",
            "Color_429",
            "Color_430",
            "Color_431",
            "Color_432",
            "Color_433",
            "Color_434",
            "Color_435",
            "Color_436",
            "Color_437",
            "Color_438",
            "Color_439",
            "Color_440",
            "Color_441",
            "Color_442",
            "Color_443",
            "Color_444",
            "Color_445",
            "Color_446",
            "Color_447",
            "Color_448",
            "Color_449",
            "Color_450",
            "Color_451",
            "Color_452",
            "Color_453",
            "Color_454",
            "Color_455",
            "Color_456",
            "Color_457",
            "Color_458",
            "Color_459",
            "Color_460",
            "Color_461",
            "Color_462",
            "Color_463",
            "Color_464",
            "Color_465",
            "Color_466",
            "Color_467",
            "Color_468",
            "Color_469",
            "Color_470",
            "Color_471",
            "Color_472",
            "Color_473",
            "Color_474",
            "Color_475",
            "Color_476",
            "Color_477",
            "Color_478",
            "Color_479",
            "Color_480",
            "Color_481",
            "Color_482",
            "Color_483",
            "Color_484",
            "Color_485",
            "Color_486",
            "Color_487",
            "Color_488",
            "Color_489",
            "Color_490",
            "Color_491",
            "Color_492",
            "Color_493",
            "Color_494",
            "Color_495",
            "Color_496",
            "Color_497",
            "Color_498",
            "Color_499",
            "Color_500",
            "Color_501",
            "Color_502",
            "Color_503",
            "Color_504",
            "Color_505",
            "Color_506",
            "Color_507",
            "Color_508",
            "Color_509",
            "Color_509",
            "Color_510",
            "Color_511",
            "Textura_LBP",
            "Textura_GLCM_Contrast",
            "Forma_Area",
            "Forma_Perimetro",
            "Forma_Circularidad",
        ]

        # Crear un DataFrame con los nombres de las características
        features_df = pd.DataFrame([features])

        # Realizar la predicción con el DataFrame
        prediction = app.config["MODEL"].predict(features_df)
        # Convertir la predicción a un tipo serializable (como int)
        prediction = int(prediction[0])
        return prediction
    else:
        print("No se pudieron extraer características de la imagen.")
        return None


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Verificar si se envió un archivo
        if "file" not in request.files:
            return jsonify({"error": "No file part in the request"}), 400

        file = request.files["file"]

        # Verificar que el archivo tenga una extensión permitida
        if file.filename == "" or not allowed_file(file.filename):
            return jsonify({"error": "Invalid or missing file name"}), 400

        # Crear el directorio de subida si no existe
        if not os.path.exists(app.config["UPLOAD_FOLDER"]):
            os.makedirs(app.config["UPLOAD_FOLDER"])

        # Guardar el archivo temporalmente
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(file_path)

        # Registrar que el archivo fue guardado
        app.logger.info(f"Archivo guardado temporalmente en: {file_path}")

        # Realizar la predicción
        prediction = predict_image_class(file_path)

        if prediction is not None:
            # Opcional: borrar el archivo después de procesarlo
            os.remove(file_path)
            return jsonify({"prediction": prediction}), 200
        else:
            return jsonify({"error": "Prediction failed"}), 500

    except Exception as e:
        # Capturar errores inesperados y registrar para depuración
        app.logger.error(f"Error durante la predicción: {str(e)}")
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500


# Iniciar el servidor de Flask
if __name__ == "__main__":
    if not os.path.exists(app.config["UPLOAD_FOLDER"]):
        os.makedirs(app.config["UPLOAD_FOLDER"])
    app.run(host="0.0.0.0", port=5000, debug=True)
