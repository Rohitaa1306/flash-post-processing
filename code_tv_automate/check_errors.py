from glob import iglob

data_folder = ""

warnings = [
    "Corrupt JPEG data",
    "DeprecationWarning",
    "UserWarning",
    "Deprecated in NumPy 1.20",
    "Failed to load image Python extension",
    "Overload resolution failed:",
    "M is not a numpy array, neither a scalar",
    "Expected Ptr<cv::UMat> for argument",
    "Traceback",
    "warpAffine",
    "nimg = face_align.norm_crop(face_img_bgr, pts5)",
    "facen = model.get_input(face, facelmarks.astype(np.int).reshape(1,5,2), face=True)",
    "face = io.imread(os.path.join(path, fname))",
    "detFacesLog, bboxFaces, idxFaces = pipe_frames_data_to_faces",
    "test_vid_frames_batch_v7_2fps_frminp_newfv_rotate.py",
    "insightface/deploy/face_model.py",
    "insightface/utils/face_align.py",
]

# Add normal strings to this list
normals = [
    "Loading symbol saved by previous version",
    "Symbol successfully upgraded!",
    "Running performance tests",
    "Resource temporarily unavailable",
    "RTNETLINK answers: File exists",
]

valid_log_paths = list(
            iglob(
                f"{data_folder}/**/{participant_id}*_logstderr.log",
                recursive=True,
            )
        )

logs_with_issues = []
for log_path in valid_log_paths:

    with open(log_path) as log:

        log_lines = log.readlines()

    issues = [
        f"{log_line.strip()}\n"
        for log_line in log_lines
        if log_line
        and all(
            normal_line not in log_line
            for normal_line in normals
        )
    ]

    warnings = [
        issue
        for issue in issues
        if any(
            warning_line in issues
            for warning_line in warnings)
    ]

    errors = [
        issue 
        for issue in issues
        if issue not in warnings
    ]

    #if warnings or errors:
    if errors:
        logs_with_issues.append(log_path)

print("\n\n".join(logs_with_issues))