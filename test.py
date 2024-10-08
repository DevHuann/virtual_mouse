import cv2
import time
import numpy as np
import pyautogui
import autopy
from PIL import Image, ImageDraw, ImageFont
import mediapipe as mp
import math

class DieuKhienCuChi:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.resize_w = 960
        self.resize_h = 720
        self.frame_margin = 100
        self.screen_width, self.screen_height = pyautogui.size()
        self.buocX, self.buocY = 0, 0
        self.toaDoCuoiX, self.toaDoCuoiY = 0, 0
        self.smoothening = 7
        self.thoiGianKichHoat = {
            'click_don': 0,
            'click_kep': 0,
            'click_phai': 0
        }
        self.mouseDown = False
        self.xuLyTay = XuLyTay()

    def nhanDien(self):
        thoiGianFPS = time.time()
        while self.cap.isOpened():
            success, anh = self.cap.read()
            if not success:
                print("Không thể chụp ảnh")
                continue

            anh = cv2.resize(anh, (self.resize_w, self.resize_h))

            anh.flags.writeable = False
            anh = cv2.cvtColor(anh, cv2.COLOR_BGR2RGB)
            anh = cv2.flip(anh, 1)
            anh = self.xuLyTay.xuLyMotBanTay(anh)
            cv2.rectangle(anh, (self.frame_margin, self.frame_margin),
                          (self.resize_w - self.frame_margin, self.resize_h - self.frame_margin), (255, 0, 255), 2)

            anh, hanhDong, diemChuChot = self.xuLyTay.kiemTraHanhDongTay(anh, ve_diem_ngon_tay=True)

            if diemChuChot:
                x3 = np.interp(diemChuChot[0], (self.frame_margin, self.resize_w - self.frame_margin),
                               (0, self.screen_width))
                y3 = np.interp(diemChuChot[1], (self.frame_margin, self.resize_h - self.frame_margin),
                               (0, self.screen_height))

                self.toaDoCuoiX = self.buocX + (x3 - self.buocX) / self.smoothening
                self.toaDoCuoiY = self.buocY + (y3 - self.buocY) / self.smoothening

                hienTai = time.time()

                if hanhDong == 'keo':
                    if not self.mouseDown:
                        pyautogui.mouseDown(button='left')
                        self.mouseDown = True
                    try:
                        if 0 <= self.toaDoCuoiX <= self.screen_width and 0 <= self.toaDoCuoiY <= self.screen_height:
                            autopy.mouse.move(self.toaDoCuoiX, self.toaDoCuoiY)
                        else:
                            print("Lỗi: Điểm nằm ngoài biên màn hình")
                    except ValueError as e:
                        print("Lỗi:", e)
                else:
                    if self.mouseDown:
                        pyautogui.mouseUp(button='left')
                    self.mouseDown = False

                if hanhDong == 'di_chuyen':
                    try:
                        if 0 <= self.toaDoCuoiX <= self.screen_width and 0 <= self.toaDoCuoiY <= self.screen_height:
                            autopy.mouse.move(self.toaDoCuoiX, self.toaDoCuoiY)
                        else:
                            print("Lỗi: Điểm nằm ngoài biên màn hình")
                    except ValueError as e:
                        print("Lỗi:", e)

                elif hanhDong == 'click_don_kich_hoat':
                    pass
                elif hanhDong == 'click_don_san_sang' and (hienTai - self.thoiGianKichHoat['click_don'] > 0.3):
                    pyautogui.click()
                    self.thoiGianKichHoat['click_don'] = hienTai
                elif hanhDong == 'click_phai_kich_hoat':
                    pass
                elif hanhDong == 'click_phai_san_sang' and (hienTai - self.thoiGianKichHoat['click_phai'] > 2):
                    pyautogui.click(button='right')
                    self.thoiGianKichHoat['click_phai'] = hienTai
                elif hanhDong == 'cuon_len':
                    pyautogui.scroll(30)
                elif hanhDong == 'cuon_xuong':
                    pyautogui.scroll(-30)

                self.buocX, self.buocY = self.toaDoCuoiX, self.toaDoCuoiY

            anh.flags.writeable = True
            anh = cv2.cvtColor(anh, cv2.COLOR_RGB2BGR)
            cThoiGian = time.time()
            fps_text = 1 / (cThoiGian - thoiGianFPS)
            thoiGianFPS = cThoiGian
            cv2.putText(anh, f"FPS: {int(fps_text)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

            cv2.imshow('Chuột ảo', anh)
            if cv2.waitKey(5) & 0xFF == 27:
                break

        self.cap.release()
        cv2.destroyAllWindows()


class XuLyTay:
    def __init__(self):
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.7,
                                          min_tracking_confidence=0.5, max_num_hands=1)
        self.danh_sach_diem_landmark = []
        self.nhan_hanh_dong = {
            'khong': 'Không',
            'di_chuyen': 'Di chuyển chuột',
            'click_don_kich_hoat': 'Nhấp chuột đơn kích hoạt',
            'click_don_san_sang': 'Nhấp chuột đơn sẵn sàng',
            'click_phai_kich_hoat': 'Nhấp chuột phải kích hoạt',
            'click_phai_san_sang': 'Nhấp chuột phải sẵn sàng',
            'cuon_len': 'Cuộn lên',
            'cuon_xuong': 'Cuộn xuống',
            'keo': 'Kéo chuột'
        }
        self.hanh_dong_phat_hien = ''

    def xuLyMotBanTay(self, anh):
        ket_qua = self.hands.process(anh)
        self.danh_sach_diem_landmark = []
        if ket_qua.multi_hand_landmarks:
            for hand_landmarks in ket_qua.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    anh,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style())
                for id_diem_landmark, truc_ngon_tay in enumerate(hand_landmarks.landmark):
                    h, w, c = anh.shape
                    toa_do_x, toa_do_y = math.ceil(truc_ngon_tay.x * w), math.ceil(truc_ngon_tay.y * h)
                    self.danh_sach_diem_landmark.append([
                        id_diem_landmark, toa_do_x, toa_do_y,
                        truc_ngon_tay.z
                    ])

                x_min, x_max = min(self.danh_sach_diem_landmark, key=lambda i: i[1])[1], \
                    max(self.danh_sach_diem_landmark, key=lambda i: i[1])[1]
                y_min, y_max = min(self.danh_sach_diem_landmark, key=lambda i: i[2])[2], \
                    max(self.danh_sach_diem_landmark, key=lambda i: i[2])[2]

                anh = cv2.rectangle(anh, (x_min - 30, y_min - 30), (x_max + 30, y_max + 30), (0, 255, 0), 2)
                anh = self.them_chu_thong_tin_viet_nam(anh, self.hanh_dong_phat_hien, (x_min - 20, y_min - 120),
                                                       (255, 0, 255), 60)

        return anh

    def them_chu_thong_tin_viet_nam(self, anh, chuoi_chu, vi_tri, mau_chu=(0, 255, 0), font_size=30):
        if isinstance(anh, np.ndarray):
            anh = Image.fromarray(cv2.cvtColor(anh, cv2.COLOR_BGR2RGB))
        ve = ImageDraw.Draw(anh)
        font_style = ImageFont.truetype("./fonts/your_vietnamese_font.ttf", font_size, encoding="utf-8")
        ve.text(vi_tri, chuoi_chu, mau_chu, font=font_style)
        return cv2.cvtColor(np.asarray(anh), cv2.COLOR_RGB2BGR)

    def tinh_khoang_cach(self, diemA, diemB):
        return math.hypot((diemA[0] - diemB[0]), (diemA[1] - diemB[1]))

    def kiemTraHanhDongTay(self, anh, ve_diem_ngon_tay=True):
        up_list = self.kiemTraNgonTayDungLen()
        hanh_dong = 'khong'

        if len(up_list) == 0:
            return anh, hanh_dong, None

        gioi_han_khoang_cach = 100
        diem_chu_chot = self.layToaDoNgonTay(8)

        if up_list == [0, 1, 0, 0, 0]:
            hanh_dong = 'di_chuyen'

        elif up_list == [1, 1, 0, 0, 0]:
            l1 = self.tinh_khoang_cach(self.layToaDoNgonTay(4), self.layToaDoNgonTay(8))
            hanh_dong = 'click_don_kich_hoat' if l1 < gioi_han_khoang_cach else 'click_don_san_sang'

        elif up_list == [0, 1, 1, 0, 0]:
            l1 = self.tinh_khoang_cach(self.layToaDoNgonTay(8), self.layToaDoNgonTay(12))
            hanh_dong = 'click_phai_kich_hoat' if l1 < gioi_han_khoang_cach else 'click_phai_san_sang'

        elif up_list == [1, 1, 1, 1, 1]:
            hanh_dong = 'cuon_len'

        elif up_list == [0, 1, 1, 1, 1]:
            hanh_dong = 'cuon_xuong'

        elif up_list == [0, 0, 1, 1, 1]:
            diem_chu_chot = self.layToaDoNgonTay(12)
            hanh_dong = 'keo'

        anh = self.ve_thong_tin(anh, hanh_dong) if ve_diem_ngon_tay else anh

        self.hanh_dong_phat_hien = self.nhan_hanh_dong[hanh_dong]

        return anh, hanh_dong, diem_chu_chot

    def layToaDoNgonTay(self, index):
        return self.danh_sach_diem_landmark[index][1], self.danh_sach_diem_landmark[index][2]

    def kiemTraNgonTayDungLen(self):
        chi_so_diem_ngon_tay = [4, 8, 12, 16, 20]
        up_list = []
        if len(self.danh_sach_diem_landmark) == 0:
            return up_list

        if self.danh_sach_diem_landmark[chi_so_diem_ngon_tay[0]][1] < self.danh_sach_diem_landmark[
            chi_so_diem_ngon_tay[0] - 1][1]:
            up_list.append(1)
        else:
            up_list.append(0)

        for i in range(1, 5):
            if self.danh_sach_diem_landmark[chi_so_diem_ngon_tay[i]][2] < self.danh_sach_diem_landmark[
                chi_so_diem_ngon_tay[i] - 2][2]:
                up_list.append(1)
            else:
                up_list.append(0)

        return up_list

    def ve_thong_tin(self, anh, hanh_dong):
        diem_ngon_tro, diem_ngon_tro, diem_ngon_giua = map(self.layToaDoNgonTay, [4, 8, 12])

        if hanh_dong == 'di_chuyen':
            anh = cv2.circle(anh, diem_ngon_tro, 20, (255, 0, 255), -1)

        elif hanh_dong == 'click_don_kich_hoat':
            diem_giua = int((diem_ngon_tro[0] + diem_ngon_tro[0]) / 2), int((diem_ngon_tro[1] + diem_ngon_tro[1]) / 2)
            anh = cv2.circle(anh, diem_giua, 30, (0, 255, 0), -1)

        elif hanh_dong == 'click_don_san_sang':
            anh = cv2.circle(anh, diem_ngon_tro, 20, (255, 0, 255), -1)
            anh = cv2.circle(anh, diem_ngon_tro, 20, (255, 0, 255), -1)
            anh = cv2.line(anh, diem_ngon_tro, diem_ngon_tro, (255, 0, 255), 2)

        elif hanh_dong == 'click_phai_kich_hoat':
            diem_giua = int((diem_ngon_tro[0] + diem_ngon_giua[0]) / 2), int((diem_ngon_tro[1] + diem_ngon_giua[1]) / 2)
            anh = cv2.circle(anh, diem_giua, 30, (0, 255, 0), -1)

        elif hanh_dong == 'click_phai_san_sang':
            anh = cv2.circle(anh, diem_ngon_tro, 20, (255, 0, 255), -1)
            anh = cv2.circle(anh, diem_ngon_giua, 20, (255, 0, 255), -1)
            anh = cv2.line(anh, diem_ngon_tro, diem_ngon_giua, (255, 0, 255), 2)
        return anh


if __name__ == "__main__":
    dieu_khien_cu_chi = DieuKhienCuChi()
    dieu_khien_cu_chi.nhanDien()


            # 'none': 'Không',
            # 'move': 'di chuyển chuột',
            # 'click_single_active': 'Nhấp chuột kích hoạt',
            # 'click_single_ready': 'Nhấp chuột sẵn sàng',
            # 'click_right_active': 'Nhấp chuột phải kích hoạt',
            # 'click_right_ready': 'Nhấp chuột phải sẵn sàng',
            # 'scroll_up': 'Cuộn lên',
            # 'scroll_down': 'Cuộn xuống',
            # 'drag': 'Kéo chuột'