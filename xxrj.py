import os
from fastapi import FastAPI, APIRouter, Depends, HTTPException
from sqlalchemy import create_engine, Column, Integer, String, ForeignKey, Float, func
from sqlalchemy.orm import declarative_base, relationship, Session, sessionmaker
from typing import List, Dict, Tuple
from pydantic import BaseModel
from sklearn.metrics.pairwise import cosine_similarity

# 创建FastAPI应用
app = FastAPI()

# ====================
# 数据库配置
# ====================
DATABASE_URL = "sqlite:///learning.db"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


# ====================
# 数据模型定义
# ====================
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    learning_records = relationship("LearningRecord", back_populates="user")


class Course(Base):
    __tablename__ = "courses"
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, index=True)
    category = Column(String)
    learning_records = relationship("LearningRecord", back_populates="course")


class LearningRecord(Base):
    __tablename__ = "learning_records"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    course_id = Column(Integer, ForeignKey("courses.id"))
    progress = Column(Float)
    user = relationship("User", back_populates="learning_records")
    course = relationship("Course", back_populates="learning_records")


Base.metadata.create_all(bind=engine)


# ====================
# 依赖项
# ====================
def get_db() -> Session:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ====================
# Pydantic模型
# ====================
class UserCreate(BaseModel):
    username: str


class CourseCreate(BaseModel):
    title: str
    category: str


class RecordCreate(BaseModel):
    user_id: int
    course_id: int
    progress: float


# ====================
# API路由
# ====================
router = APIRouter()


# 用户相关路由
@router.post("/users/")
async def create_user(user: UserCreate, db: Session = Depends(get_db)):
    new_user = User(username=user.username)
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return {"user_id": new_user.id, "username": new_user.username}


# 课程相关路由
@router.post("/courses/")
async def create_course(course: CourseCreate, db: Session = Depends(get_db)):
    new_course = Course(title=course.title, category=course.category)
    db.add(new_course)
    db.commit()
    db.refresh(new_course)
    return {"course_id": new_course.id, "title": new_course.title}


# 学习记录相关路由
@router.post("/records/")
async def create_record(record: RecordCreate, db: Session = Depends(get_db)):
    # 验证用户存在
    user = db.query(User).get(record.user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    # 验证课程存在
    course = db.query(Course).get(record.course_id)
    if not course:
        raise HTTPException(status_code=404, detail="Course not found")

    new_record = LearningRecord(
        user_id=record.user_id,
        course_id=record.course_id,
        progress=record.progress
    )
    db.add(new_record)
    db.commit()
    db.refresh(new_record)
    return {"record_id": new_record.id}


# 推荐系统路由
@router.get("/users/{user_id}/recommendations")
async def get_recommendations(user_id: int, db: Session = Depends(get_db)):
    records = db.query(LearningRecord).filter_by(user_id=user_id).all()

    if not records:
        # 获取热门课程（按学习人数排序）
        popular_courses = db.query(
            Course.id,
            func.count(LearningRecord.id).label('course_count')
        ).join(
            LearningRecord, Course.id == LearningRecord.course_id
        ).group_by(Course.id).order_by(func.count(LearningRecord.id).desc()).limit(5).all()

        return {"recommended_courses": [db.query(Course).get(course[0]).title for course in popular_courses]}

    recommendations = recommend_courses(user_id, records, db)
    return {"recommended_courses": [course.title for course in recommendations]}


# ====================
# 推荐算法
# ====================
def recommend_courses(user_id: int, records: List[LearningRecord], db: Session) -> List[Course]:
    similar_users = get_similar_users(user_id, records, db)

    candidate_courses = set()
    for similar_user_id in similar_users:
        user_records = db.query(LearningRecord).filter_by(user_id=similar_user_id).all()
        for record in user_records:
            candidate_courses.add(record.course_id)

    learned_courses = {record.course_id for record in records}
    candidate_courses -= learned_courses

    if not candidate_courses:
        return []

    # 计算课程热度
    course_popularity = {}
    for course_id in candidate_courses:
        course_popularity[course_id] = db.query(LearningRecord).filter_by(course_id=course_id).count()

    # 按热度排序
    sorted_courses = sorted(course_popularity.items(), key=lambda x: x[1], reverse=True)
    recommended_course_ids = [course_id for course_id, _ in sorted_courses[:5]]

    return db.query(Course).filter(Course.id.in_(recommended_course_ids)).all()


def get_similar_users(user_id: int, records: List[LearningRecord], db: Session) -> List[int]:
    target_records = {r.course_id: r.progress for r in records}

    # 构建用户-课程矩阵
    all_users = {}
    for record in db.query(LearningRecord):
        user_courses = all_users.setdefault(record.user_id, {})
        user_courses[record.course_id] = record.progress

    user_ids = list(all_users.keys())
    if not user_ids:
        return []

    # 创建特征矩阵
    user_matrix = []
    for uid in user_ids:
        vector = []
        for cid in target_records.keys():
            vector.append(all_users[uid].get(cid, 0))
        user_matrix.append(vector)

    # 计算余弦相似度
    try:
        similarities = cosine_similarity([list(target_records.values())], user_matrix)
    except ValueError:
        return []

    # 排除当前用户自己
    similar_users = []
    for i, uid in enumerate(user_ids):
        if uid != user_id:
            similar_users.append((uid, similarities[0][i]))

    # 按相似度排序取前三
    similar_users.sort(key=lambda x: x[1], reverse=True)
    return [uid for uid, _ in similar_users[:3]]


# ====================
# 注册路由
# ====================
app.include_router(router)