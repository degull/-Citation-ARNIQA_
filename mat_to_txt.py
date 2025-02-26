import scipy.io
import numpy as np
import os

def mat_to_txt(mat_file, txt_file):
    # .mat 파일 로드
    data = scipy.io.loadmat(mat_file)

    with open(txt_file, 'w', encoding='utf-8') as f:
        for key, value in data.items():
            # MATLAB의 기본 변수 필터링
            if key.startswith("__"):
                continue
            f.write(f"Variable: {key}\n")

            # 데이터 타입 확인 후 저장
            if isinstance(value, np.ndarray):
                if value.dtype == np.object_:  # 문자열 배열 처리
                    for row in value:
                        f.write(" ".join(map(str, row)) + "\n")
                else:  # 숫자 배열 처리
                    np.savetxt(f, value, fmt='%g')
            elif isinstance(value, (list, tuple)):  # 리스트, 튜플 처리
                f.write(" ".join(map(str, value)) + "\n")
            else:  # 단일 값 처리
                f.write(str(value) + "\n")

            f.write("\n")

    print(f"Saved {mat_file} as {txt_file}")

# .mat 파일 경로 설정
mat_file_path = r"E:\ARNIQA - SE - mix\ARNIQA\dataset\LIVE\refnames_all.mat"
output_txt_path = os.path.splitext(mat_file_path)[0] + ".txt"  # 같은 경로에 .txt 저장

# 변환 실행
mat_to_txt(mat_file_path, output_txt_path)
