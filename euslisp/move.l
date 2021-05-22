#!/usr/bin/env roseus

;; load robot, room and arrow
;;(load "package://pr2eus/pr2.l")   
(load "models/room73b2-scene.l")
(load "models/arrow-object.l")
(require :pr2-interface "package://pr2eus/pr2-interface.l")
(require :eng2-scene "package://jsk_maps/src/eng2-scene.l")
(ros::roseus "pr2_move_node")

;; set spots
;;(setq *room73b2* (room73b2))
;;(setq *cook-spot* (send *room73b2* :spot "cook-spot"))
;;(setq *sink-spot* (make-coords :pos #f(1055 2600 0) :rpy #f(3.14 0 0))) 
;;(setq *kitchen-spot* (send *room73b2* :spot "kitchen-spot"))

(defun init()
  ;; set room73b2
  ;;(setq *room73b2* (room73b2))

  ;; init 73B2 room
  (unless (boundp '*scene*) (setq *scene* (make-eng2-scene)))

  ;; init PR2
  (unless (boundp '*pr2*) (setq *pr2* (instance pr2-sensor-robot :init)))
  (unless (boundp '*ri*) (setq *ri* (instance pr2-interface :init)))
  (send *pr2* :move-to (send *ri* :state :worldcoords) :world)
  
  ;; set arrow
  (setq *axis* (arrow))
  
  ;; IRT viewer
  ;;(objects (list *pr2* *room73b2* *axis*))
  (objects (list *pr2* *scene* *axis*))
  (send *irtviewer* :look-all *pr2*)
  ;; set cup
  ;;(setq *cup* (send *room73b2* :object "room73b2-mug-cup"))
  ;;(if (send *cup* :parent) (send (send *cup* :parent) :dissoc *cup*))
  ;;;; set dish
  ;;(setq *dish* (send *room73b2* :object "room73b2-dish"))
  ;;(if (send *dish* :parent) (send (send *dish* :parent) :dissoc *dish*))
  ;;(format t ";; please run (main)~%")
  )

(defun go-to-sink ()
  (send *pr2* :move-to (send *room73b2* :spot "/eng2/7f/room73B2-sink-front1")  :world)
  (send *pr2* :move-to (make-coords :pos #f(300 -150 0)))
  (send *ri* :angle-vector (send *pr2* :angle-vector) 4000)
  (send *ri* :wait-interpolation)
  )

;; キッチンの場所まで移動 (初期値)
(defun move-to-kitchen-irt ()
    ;;(send *pr2* :move-to *kitchen-spot* :world)
    )

(defun move-to-sink-irt ()
  (send *pr2* :move-to *sink-spot* :world)
  )

(defun move-to-sink-ri ()
  ;;(send *ri* :move-to *sink-spot*)
  ;; (go-to-spot "/eng2/7f/room73B2-sink-front")
  (send *ri* :move-to (send *scene* :spot "/eng2/7f/room73B2-sink-front1") :frame-id "world")
  )

(defun move-a-little ()
  ;; move forward 0.12 and move right -0.14
  ;; but need to be adjust more
  (send *ri* :go-pos-unsafe 0.12 -0.14 0)
)

;;*irtviewer*上のロボットを拡大表示
(defun view-adj ()
  (send *irtviewer* :look-all
	(geo::make-bounding-box
	 (flatten (send-all (send *pr2* :bodies) :vertices))))
  )

(defun main ()
  (pr2-init)
  (init)
  ;;(go-to-sink)
  (move-to-sink-ri)
  (move-a-littele)
  (view-adj)
  )

(main)
