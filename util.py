import sys
import cv2
import numpy as np
from thundernet.utils.np_opr import calc_iou
from thundernet.utils.np_opr import rpn_to_roi, non_max_suppression_fast, apply_regr
from model_evaluate_tool import voc_mAP


def get_img_output_length(width, height):
    def get_output_length(input_length):
        return input_length//16

    return get_output_length(width), get_output_length(height)


def format_img_size(img, C):
    """ formats the image size based on config """
    (height, width, _) = img.shape
    print(height, width)
    ratio_h = height / 320
    ratio_w = width / 320
    img = cv2.resize(img, (320, 320), interpolation=cv2.INTER_CUBIC)
    return img, ratio_h, ratio_w


def format_img_channels(img, C):
    """ formats the image channels based on config """
    img = img[:, :, (2, 1, 0)]      # RGB
    img = img.astype(np.float32)
    img[:, :, 0] -= C.img_channel_mean[0]
    img[:, :, 1] -= C.img_channel_mean[1]
    img[:, :, 2] -= C.img_channel_mean[2]
    img /= C.img_scaling_factor
    img = np.transpose(img, (2, 0, 1))      # (channels, h, w)
    img = np.expand_dims(img, axis=0)
    return img


def format_img(img, C):
    """ formats an image for model prediction based on config """
    img, ratio_h, ratio_w = format_img_size(img, C)
    img = format_img_channels(img, C)
    return img, ratio_h, ratio_w


# Method to transform the coordinates of the bounding box to its original size
def get_real_coordinates(ratio_h, ratio_w, x1, y1, x2, y2):
    real_x1 = int(round(x1 * ratio_w))
    real_y1 = int(round(y1 * ratio_h))
    real_x2 = int(round(x2 * ratio_w))
    real_y2 = int(round(y2 * ratio_h))

    return (real_x1, real_y1, real_x2 ,real_y2)


def get_data(input_path):
    """Parse the data from annotation file

    Args:
        input_path: annotation file path

    Returns:
        all_data: list(filepath, width, height, list(bboxes))
        classes_count: dict{key:class_name, value:count_num}
            e.g. {'Car': 2383, 'Mobile phone': 1108, 'Person': 3745}
        class_mapping: dict{key:class_name, value: idx}
            e.g. {'Car': 0, 'Mobile phone': 1, 'Person': 2}
    """
    found_bg = False
    all_imgs = {}

    classes_count = {}

    class_mapping = {}

    i = 1

    with open(input_path, 'r') as f:

        print('Parsing annotation files')

        for line in f:

            # Print process
            sys.stdout.write('\r' + 'idx=' + str(i))
            i += 1

            # line_split = line.strip().split(',')

            # (filename, y1, x1, y2, x2, class_name) = line_split
            print(line)
            filename = line.split()[0]
            y1, x1, y2, x2, class_name = line.split()[1].split(',')

            if class_name == '':
                # wrong labelling
                continue

            if class_name not in classes_count:
                classes_count[class_name] = 1
            else:
                classes_count[class_name] += 1

            if class_name not in class_mapping:
                if class_name == 'bg' and found_bg == False:
                    print(
                        'Found class name bg.Will be treated as background (this is usually for hard negative mining).')
                    found_bg = True
                class_mapping[class_name] = len(class_mapping)

            if filename not in all_imgs:
                all_imgs[filename] = {}

                img = cv2.imread(filename)
                (rows, cols) = img.shape[:2]
                all_imgs[filename]['filepath'] = filename
                all_imgs[filename]['width'] = cols
                all_imgs[filename]['height'] = rows
                all_imgs[filename]['bboxes'] = []
            # if np.random.randint(0,6) > 0:
            # 	all_imgs[filename]['imageset'] = 'trainval'
            # else:
            # 	all_imgs[filename]['imageset'] = 'test'

            all_imgs[filename]['bboxes'].append(
                {'class': class_name, 'x1': int(x1), 'x2': int(x2), 'y1': int(y1), 'y2': int(y2)})

        all_data = []
        for key in all_imgs:
            all_data.append(all_imgs[key])

        # make sure the bg class is last in the list
        if found_bg:
            if class_mapping['bg'] != len(class_mapping) - 1:
                key_to_switch = [key for key in class_mapping.keys() if class_mapping[key] == len(class_mapping) - 1][0]
                val_to_switch = class_mapping['bg']
                class_mapping['bg'] = len(class_mapping) - 1
                class_mapping[key_to_switch] = val_to_switch

        return all_data, classes_count, class_mapping


# def validate_model_(model_rpn, model_classifier, data_gen_val, C, val_samples, val_batch, *params):
#     # validate model
#     # val_batch = 20
#     val_num_rois = C.num_rois  # 每组处理roi数训练验证应保持一致
#     iterators = val_samples//val_batch
#     loss_val = np.zeros((iterators, 5))
#     for i in range(iterators):
#         imgs = []
#         rpn_label = [[], []]
#         img_datas = []
#         for j in range(val_batch):
#             X, Y, img_data, debug_img, debug_num_pos = next(data_gen_val)
#             imgs.append(X)
#             rpn_label[0].append(Y[0])
#             rpn_label[1].append(Y[1])
#             img_datas.append(img_data)
#         imgs = np.vstack(imgs)
#         rpn_label[0] = np.vstack(rpn_label[0])
#         rpn_label[1] = np.vstack(rpn_label[1])
#         loss_rpn_val = model_rpn.test_on_batch(imgs, rpn_label)
#
#         rpn_val = model_rpn.predict_on_batch(imgs)
#         loss_cls = np.zeros((4, ))
#         for img_idx in range(val_batch):
#             # gen data for classifier
#             R = rpn_to_roi(np.expand_dims(rpn_val[0][img_idx, :, :, :], axis=0),
#                            np.expand_dims((rpn_val[1])[img_idx, :, :, :], axis=0),
#                            C, 'tf', use_regr=True, overlap_thresh=0.7, max_boxes=300)
#             X2, Y1, Y2, IouS = calc_iou(R, img_datas[img_idx], C)
#
#             if X2 is None:
#                 continue
#
#             neg_samples_val = np.where(Y1[0, :, -1] == 1)  # index of class 'bg' is len(class_mapping)
#             pos_samples_val = np.where(Y1[0, :, -1] == 0)
#             if len(neg_samples_val) > 0:
#                 neg_samples_val = neg_samples_val[0]
#             else:
#                 neg_samples_val = []
#
#             if len(pos_samples_val) > 0:
#                 pos_samples_val = pos_samples_val[0]
#             else:
#                 pos_samples_val = []
#             if len(pos_samples_val) < val_num_rois // 2:
#                 selected_pos_samples_val = pos_samples_val.tolist()
#             else:
#                 selected_pos_samples_val = np.random.choice(pos_samples_val, val_num_rois // 2,
#                                                             replace=False).tolist()
#
#             # Randomly choose (num_rois - num_pos) neg samples
#             try:
#                 selected_neg_samples_val = np.random.choice(neg_samples_val,
#                                                             val_num_rois - len(selected_pos_samples_val),
#                                                             replace=False).tolist()
#             except ValueError:
#                 try:
#                     selected_neg_samples = np.random.choice(neg_samples_val, C.num_rois - len(selected_pos_samples_val),
#                                                             replace=True).tolist()
#                 except:
#                     # jump through
#                     continue
#
#             # Save all the pos and neg samples in sel_samples
#             sel_samples_val = selected_pos_samples_val + selected_neg_samples_val
#             loss_cls += model_classifier.test_on_batch([np.expand_dims(imgs[img_idx], 0), X2[:, sel_samples_val, :]],
#                                                           [Y1[:, sel_samples_val, :], Y2[:, sel_samples_val, :]])
#
#             '''loss_class_val = model_classifier.test_on_batch([imgs[img_idx], X2[:, sel_samples_val, :]],
#                                                             [Y1[:, sel_samples_val, :],
#                                                              Y2[:, sel_samples_val, :]])'''
#         loss_cls /= val_batch
#
#         loss_val[i, 0] = loss_rpn_val[1]
#         loss_val[i, 1] = loss_rpn_val[2]
#
#         loss_val[i, 2] = loss_cls[1]
#         loss_val[i, 3] = loss_cls[2]
#         loss_val[i, 4] = loss_cls[3]
#     loss_rpn_cls_val = np.mean(loss_val[:, 0])
#     loss_rpn_regr_val = np.mean(loss_val[:, 1])
#     loss_class_cls_val = np.mean(loss_val[:, 2])
#     loss_class_regr_val = np.mean(loss_val[:, 3])
#     acc_class_val = np.mean(loss_val[:, 4])
#
#     return loss_rpn_cls_val, loss_rpn_regr_val, loss_class_cls_val, loss_class_regr_val, acc_class_val


def validate_model(model_rpn, model_classifier, data_gen_val, C, val_samples, val_batch, bbox_threshold=0.5, overlap_thresh=0.7, **params):
    iterators = val_samples//val_batch
    class_mapping = C.class_mapping
    class_mapping = {v: k for k, v in class_mapping.items()}
    # class_to_color = {class_mapping[v]: np.random.randint(0, 255, 3) for v in class_mapping}
    # output_path = './data/{}'

    loss_val = np.zeros((iterators, 5))
    bbox_result = list()        # bboxes result
    img_datas = list()      # ground truth data

    for i in range(iterators):
        imgs = []
        rpn_label = [[], []]
        for j in range(val_batch):
            X, Y, img_data, _, _ = next(data_gen_val)
            imgs.append(X)
            rpn_label[0].append(Y[0])
            rpn_label[1].append(Y[1])
            img_datas.append(img_data)
        imgs = np.vstack(imgs)
        rpn_label[0] = np.vstack(rpn_label[0])
        rpn_label[1] = np.vstack(rpn_label[1])
        loss_rpn_val = model_rpn.test_on_batch(imgs, rpn_label)

        loss_cls = np.zeros((4, ))
        valid_cls_img = 0     # valid test img number
        for img_idx in range(val_batch):
            try:
                # traverse all imgs in a val_batch
                img_data = img_datas[img_idx]
                img = np.expand_dims(imgs[img_idx], axis=0)
                [Y1, Y2] = model_rpn.predict(img)
                # Y1(1, 20, 20, 9)  Y2(1, 20, 20, 36)
                # gen data for classifier
                # (max_boxes, 4)
                R = rpn_to_roi(Y1, Y2, C, 'tf', use_regr=True, overlap_thresh=overlap_thresh, max_boxes=200)

                # ------ calculate loss ------
                X2, Y1, Y2, IouS = calc_iou(R, img_datas[img_idx], C)
                if X2 is None:
                    continue
                loss_cls_img = np.zeros((4, ))        # loss of one img
                for loss_roi_batch in range(X2.shape[1]//C.num_rois+1):
                    # traverse all proposed rois
                    if loss_roi_batch == X2.shape[1]//C.num_rois:
                        # no enough sample, pad
                        # rois = X2[:, C.num_rois*loss_roi_batch:, :]
                        # label_cls = Y1[:, C.num_rois*loss_roi_batch:, :]
                        # label_regr = Y2[:, C.num_rois*loss_roi_batch:, :]
                        # curr_shape = rois.shape
                        # target_shape = (curr_shape[0], C.num_rois, curr_shape[2])
                        # rois_pad = np.zeros(target_shape).astype(rois.dtype)
                        # rois_pad[:, :curr_shape[1], :] = rois
                        # rois_pad[0, curr_shape[1]:, :] = rois[0, 0:C.num_rois-curr_shape[1], :]
                        break
                    else:
                        rois = X2[:, C.num_rois*loss_roi_batch:C.num_rois*(loss_roi_batch+1), :]
                        label_cls = Y1[:, C.num_rois*loss_roi_batch:C.num_rois*(loss_roi_batch+1), :]
                        label_regr = Y2[:, C.num_rois*loss_roi_batch:C.num_rois*(loss_roi_batch+1), :]
                    loss_cls_img += model_classifier.test_on_batch([np.expand_dims(imgs[img_idx], 0), rois], [label_cls, label_regr])
                loss_cls_img = loss_cls_img / (X2.shape[1]//C.num_rois+1)
                valid_cls_img += 1
                loss_cls += loss_cls_img

                # ------ check ground truth and calculate mAP ------
                if params['evaluate_mAP'] is True:
                    # every several epochs test mAP one time
                    # convert from (x1,y1,x2,y2) to (x,y,w,h)-->20X20 feature map
                    R[:, 2] -= R[:, 0]
                    R[:, 3] -= R[:, 1]

                    # apply the spatial pyramid pooling to the proposed regions
                    bboxes = {}
                    probs = {}

                    for jk in range(R.shape[0] // C.num_rois + 1):  # 300//4+1
                        # traverse all rois rpn predicted in R (300)
                        # 每组处理roi数训练验证应保持一致
                        ROIs = np.expand_dims(R[C.num_rois * jk:C.num_rois * (jk + 1), :], axis=0)
                        if ROIs.shape[1] == 0:
                            break

                        if jk == R.shape[0] // C.num_rois:  # ???
                            # pad R
                            curr_shape = ROIs.shape
                            target_shape = (curr_shape[0], C.num_rois, curr_shape[2])
                            ROIs_padded = np.zeros(target_shape).astype(ROIs.dtype)
                            ROIs_padded[:, :curr_shape[1], :] = ROIs
                            ROIs_padded[0, curr_shape[1]:, :] = ROIs[0, 0, :]
                            ROIs = ROIs_padded

                        [P_cls, P_regr] = model_classifier.predict([img, ROIs])
                        #         print([P_cls, P_regr])
                        # Calculate bboxes coordinates on resized image
                        for ii in range(P_cls.shape[1]):  # all bboxes in batch
                            # Ignore 'bg' class
                            if np.max(P_cls[0, ii, :]) < bbox_threshold or np.argmax(P_cls[0, ii, :]) == (P_cls.shape[2] - 1):
                                # print(np.argmax(P_cls[0, ii, :]))
                                # 所有类对应可能性均小于阈值/最大值为最后一类(bg)
                                continue

                            cls_name = class_mapping[np.argmax(P_cls[0, ii, :])]

                            if cls_name not in bboxes:
                                bboxes[cls_name] = []
                                probs[cls_name] = []

                            (x, y, w, h) = ROIs[0, ii, :]

                            cls_num = np.argmax(P_cls[0, ii, :])
                            try:
                                (tx, ty, tw, th) = P_regr[0, ii, 4 * cls_num:4 * (cls_num + 1)]
                                tx /= C.classifier_regr_std[0]
                                ty /= C.classifier_regr_std[1]
                                tw /= C.classifier_regr_std[2]
                                th /= C.classifier_regr_std[3]
                                x, y, w, h = apply_regr(x, y, w, h, tx, ty, tw, th)
                            except:
                                pass
                            # bboxes : (x1, y1, x2, y2) in 320x320 image
                            bboxes[cls_name].append(
                                [C.rpn_stride * x, C.rpn_stride * y, C.rpn_stride * (x + w), C.rpn_stride * (y + h)])
                            probs[cls_name].append(np.max(P_cls[0, ii, :]))

                    if len(bboxes) == 0:
                        # no object detected, choose rois to test and calculate loss
                        continue

                    bbox_result_img = {'filepath': img_data['filepath'], 'bboxes': list()}
                    # img = np.squeeze(img, axis=0).astype('uint8')
                    for key in bboxes:  # all detected classes
                        bbox = np.array(bboxes[key])
                        # print(key)
                        new_boxes, new_probs = non_max_suppression_fast(bbox, np.array(probs[key]), overlap_thresh=overlap_thresh, max_boxes=50)
                        for jk in range(new_boxes.shape[0]):
                            (x1, y1, x2, y2) = new_boxes[jk, :]
                            # Calculate real coordinates on original image
                            ratio_h, ratio_w = img_data['height']/320, img_data['width']/320
                            (real_x1, real_y1, real_x2, real_y2) = get_real_coordinates(ratio_h, ratio_w, x1, y1, x2, y2)
                            # bbox result
                            bbox_result_img['bboxes'].append({'x1': real_x1, 'y1': real_y1,
                                                'x2': real_x2, 'y2': real_y2, 'class': '{}:{:.2f}'.format(key, new_probs[jk])})
                    #         cv2.rectangle(img, (real_x1, real_y1), (real_x2, real_y2),
                    #                       (int(class_to_color[key][0]), int(class_to_color[key][1]),
                    #                        int(class_to_color[key][2])), 2)
                    #
                    #         textLabel = '{}:{}'.format(key, int(100 * new_probs[jk]))
                    #
                    #         (retval, baseLine) = cv2.getTextSize(textLabel, cv2.FONT_HERSHEY_COMPLEX, 1, 1)
                    #         textOrg = (real_x1, real_y1)
                    #
                    #         cv2.rectangle(img, (textOrg[0] - 5, textOrg[1] + baseLine - 5),
                    #                       (textOrg[0] + retval[0] + 5, textOrg[1] - retval[1] - 5), (0, 0, 0), 1)  # bbox
                    #         cv2.rectangle(img, (textOrg[0] - 5, textOrg[1] + baseLine - 5),
                    #                       (textOrg[0] + retval[0] + 5, textOrg[1] - retval[1] - 5), (255, 255, 255),
                    #                       -1)  # text box
                    #         cv2.putText(img, textLabel, textOrg, cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 0), 1)
                    # cv2.imwrite(output_path.format(img_data['filepath']), img)
                    bbox_result.append(bbox_result_img)
            except:
                continue

        # average loss in a validation bath
        if valid_cls_img == 0:
            continue
        else:
            loss_cls /= valid_cls_img

        loss_val[i, 0] = loss_rpn_val[1]
        loss_val[i, 1] = loss_rpn_val[2]

        loss_val[i, 2] = loss_cls[1]
        loss_val[i, 3] = loss_cls[2]
        loss_val[i, 4] = loss_cls[3]
    loss_rpn_cls_val = np.mean(loss_val[:, 0])
    loss_rpn_regr_val = np.mean(loss_val[:, 1])
    loss_class_cls_val = np.mean(loss_val[:, 2])
    loss_class_regr_val = np.mean(loss_val[:, 3])
    acc_class_val = np.mean(loss_val[:, 4])

    # mAP
    if len(bbox_result) < 1 or params['evaluate_mAP'] is False:
        mAP = 0
    else:
        mAP, _ = voc_mAP(bbox_result, img_datas, mode='validate', cls_count=params['val_class_count'])
        mAP = [mAP, _[0]]

    return loss_rpn_cls_val, loss_rpn_regr_val, loss_class_cls_val, loss_class_regr_val, acc_class_val, mAP

