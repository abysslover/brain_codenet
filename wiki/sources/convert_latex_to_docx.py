'''
Created on 2025. 7. 14.

@author: "Eun-Cheon Lim @ Neuroears"
'''
import pypandoc
import os
import glob
import subprocess

def install_pandoc():
    import pypandoc
    pypandoc.download_pandoc()
    
def convert_latex_to_docx(input_tex, output_docx):
    if os.path.exists(output_docx):
        print(f"이미 존재: {output_docx}")
        return
    try:
        output = pypandoc.convert_file(
            input_tex,
            'docx',
            outputfile=output_docx,
            extra_args=['--mathjax']
        )
        print(f"변환 완료: {output_docx}")
    except Exception as e:
        print(f"변환 중 오류 발생 ({input_tex} -> {output_docx}): {e}")
        
def convert_latex_to_pdf_by_pandoc(input_tex, output_pdf):
    if os.path.exists(output_pdf):
        print(f"이미 존재: {output_pdf}")
        return
    try:
        output = pypandoc.convert_file(
            input_tex,
            'pdf',
            outputfile=output_pdf,
            extra_args=['--pdf-engine=xelatex', '--mathjax', '-V', 'CJKmainfont=Noto Serif CJK KR']
        )
        print(f"변환 완료: {output_pdf}")
    except Exception as e:
        print(f"변환 중 오류 발생 ({input_tex} -> {output_pdf}): {e}")

def convert_latex_to_pdf(input_tex, output_pdf):
    """
    LaTeX 파일을 PDF로 변환하고 성공/실패에 따라 보조 파일을 정리합니다.
    - 성공 시: 모든 보조 파일 삭제 (.aux, .log, .out, .toc 등)
    - 실패 시: .log 파일만 남기고 나머지 삭제
    """
    if os.path.exists(output_pdf):
        print(f"이미 존재: {output_pdf}")
        return True
    
    base_name = os.path.splitext(input_tex)[0]
    
    def clean_auxiliary_files(keep_log=False):
        """
        LaTeX 보조 파일을 정리합니다.
        Args:
            keep_log (bool): True면 .log 파일을 보존, False면 모든 파일 삭제
        """
        # 처리할 보조 파일 확장자 목록
        extensions = ['.aux', '.out', '.toc', '.fls', '.fdb_latexmk', '.synctex.gz']
        if not keep_log:
            extensions.append('.log')
        
        deleted_files = []
        for ext in extensions:
            aux_file = base_name + ext
            if os.path.exists(aux_file):
                try:
                    os.remove(aux_file)
                    deleted_files.append(aux_file)
                except Exception as e:
                    print(f"⚠️ 파일 삭제 실패: {aux_file} - {e}")
        
        if deleted_files:
            print(f"  정리된 파일: {', '.join([os.path.basename(f) for f in deleted_files])}")
    
    try:
        # 출력 디렉토리 생성
        output_dir = os.path.dirname(output_pdf)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # 컴파일 시작 전 기존 보조 파일 정리
        clean_auxiliary_files(keep_log=False)
        
        # XeLaTeX 2번 실행 (목차와 참조 완성)
        for run in range(2):
            result = subprocess.run(
                ['xelatex', '-interaction=nonstopmode', '-output-directory', output_dir or '.', input_tex],
                capture_output=True, text=True, check=True
            )
            print(f"XeLaTeX 실행 {run+1}/2 완료")
        
        # PDF 파일 확인 및 이동
        generated_pdf = os.path.splitext(input_tex)[0] + '.pdf'
        if os.path.exists(generated_pdf):
            if generated_pdf != output_pdf:
                os.rename(generated_pdf, output_pdf)
            print(f"✅ 변환 완료: {output_pdf}")
            
            # 성공 시: 모든 보조 파일 삭제 (log 포함)
            print("성공 - 모든 보조 파일 정리:")
            clean_auxiliary_files(keep_log=False)
            
            return True
        else:
            print(f"❌ 오류: {generated_pdf} 파일이 생성되지 않았습니다.")
            
            # 실패 시: log만 남기고 나머지 삭제
            print("실패 - 디버깅을 위해 .log 파일만 유지:")
            clean_auxiliary_files(keep_log=True)
            
            return False
            
    except subprocess.CalledProcessError as e:
        print(f"❌ XeLaTeX 실행 오류:")
        if e.stderr:
            # 중요한 오류만 출력
            error_lines = e.stderr.split('\n')
            important_errors = [line for line in error_lines 
                              if any(keyword in line for keyword in ['Error', 'Missing', 'Fatal'])]
            for error in important_errors[:5]:  # 최대 5개만 출력
                print(f"  {error.strip()}")
        
        # 실패 시: log만 남기고 나머지 삭제
        print("실패 - 디버깅을 위해 .log 파일만 유지:")
        clean_auxiliary_files(keep_log=True)
        
        return False
        
    except FileNotFoundError:
        print("❌ xelatex이 설치되지 않았습니다.")
        print("설치 방법:")
        print("  Ubuntu/Debian: sudo apt-get install texlive-xetex texlive-lang-korean")
        print("  macOS: brew install --cask mactex")
        
        # 실패 시: log만 남기고 나머지 삭제 (있다면)
        print("실패 - 디버깅을 위해 .log 파일만 유지:")
        clean_auxiliary_files(keep_log=True)
        
        return False
        
    except Exception as e:
        print(f"❌ 예상치 못한 오류: {e}")
        
        # 실패 시: log만 남기고 나머지 삭제
        print("실패 - 디버깅을 위해 .log 파일만 유지:")
        clean_auxiliary_files(keep_log=True)
        
        return False


def convert_all_tex_files():
    # 현재 디렉토리의 모든 .tex 파일 찾기
    tex_files = glob.glob("*.tex")
    if not tex_files:
        print("현재 디렉토리에 .tex 파일이 없습니다.")
        return
    
    for tex_file in tex_files:
        # 파일 이름에서 확장자 제거
        base_name = os.path.splitext(tex_file)[0]
        pdf_output = f"{base_name}.pdf"
        docx_output = f"{base_name}.docx"
        
        # PDF로 변환
        convert_latex_to_pdf(tex_file, pdf_output)
        # DOCX로 변환
        if "20250728_face_detector_yolo" in tex_file:
            convert_latex_to_docx(tex_file, docx_output)

if __name__ == "__main__":
    # Pandoc 설치 확인
    try:
        pypandoc.get_pandoc_version()
    except OSError:
        print("Pandoc이 설치되지 않았습니다. 설치 중...")
        install_pandoc()
    
    # 모든 .tex 파일 변환
    convert_all_tex_files()
        
        
        