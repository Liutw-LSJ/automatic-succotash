import os
from fastapi import FastAPI, APIRouter, Depends, HTTPException
from sqlalchemy import create_engine, Column, Integer, String, ForeignKey, Float, func, select
from sqlalchemy.orm import declarative_base, relationship, Session, sessionmaker
from sqlalchemy.exc import IntegrityError
from typing import List, Dict, Tuple, Optional
from pydantic import BaseModel, validator
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import lil_matrix
from cachetools import cached, TTLCache

# 创建FastAPI应用
app = FastAPI()

# ====================
# 数据库配置
# ====================
DATABASE_URL = "sqlite:///learning.db"
engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


# ====================
# 数据模型定义
# ====================
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, index=True, nullable=False)
    learning_records = relationship("LearningRecord", back_populates="user", cascade="all, delete-orphan")


class Course(Base):
    __tablename__ = "courses"
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String(100), index=True, nullable=False)
    category = Column(String(50))
    learning_records = relationship("LearningRecord", back_populates="course", cascade="all, delete-orphan")


class LearningRecord(Base):
    __tablename__ = "learning_records"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    course_id = Column(Integer, ForeignKey("courses.id"), nullable=False)
    progress = Column(Float, check=lambda x: x >= 0.0 and x <= 1.0)
    user = relationship("User", back_populates="learning_records")
    course = relationship("Course", back_populates="learning_records")


Base.metadata.create_all(bind=engine)

# ====================
# 缓存配置
# ====================
popular_courses_cache = TTLCache(maxsize=100, ttl=300)  # 5分钟缓存


# ====================
# 依赖项
# ====================
def get_db() -> Session:
    db = SessionLocal()
    try:
        yield db
        db.commit()  # 确保显式提交
    except:
        db.rollback()
    finally:
        db.close()


# ====================
# Pydantic模型
# ====================
class UserCreate(BaseModel):
    username: str

    @validator("username")
    def validate_username(cls, v):
        if len(v) < 3 or len(v) > 50:
            raise ValueError("Username must be 3-50 characters")
        return v


class CourseCreate(BaseModel):
    title: str
    category: Optional[str] = None

    @validator("title")
    def validate_title(cls, v):
        if len(v) < 5 or len(v) > 100:
            raise ValueError("Title must be 5-100 characters")
        return v


class RecordCreate(BaseModel):
    user_id: int
    course_id: int
    progress: float

    @validator("progress")
    def validate_progress(cls, v):
        if not (0.0 <= v <= 1.0):
            raise ValueError("Progress must be between 0.0 and 1.0")
        return v


# ====================
# API路由
# ====================
router = APIRouter()


@router.post("/users/", response_model=UserCreate)
def create_user(user: UserCreate, db: Session = Depends(get_db)):
    try:
        new_user = User(username=user.username)
        db.add(new_user)
        db.commit()
        db.refresh(new_user)
    except IntegrityError:
        raise HTTPException(status_code=400, detail="Username already exists")
    return {"user_id": new_user.id, "username": new_user.username}


@router.post("/courses/", response_model=CourseCreate)
def create_course(course: CourseCreate, db: Session = Depends(get_db)):
    try:
        new_course = Course(title=course.title, category=course.category)
        db.add(new_course)
        db.commit()
        db.refresh(new_course)
    except IntegrityError:
        raise HTTPException(status_code=400, detail="Course title already exists")
    return {"course_id": new_course.id, "title": new_course.title}


@router.post("/records/")
def create_record(record: RecordCreate, db: Session = Depends(get_db)):
    user = db.query(User).get(record.user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    course = db.query(Course).get(record.course_id)
    if not course:
        raise HTTPException(status_code=404, detail="Course not found")

    existing_record = db.query(LearningRecord).filter_by(
        user_id=record.user_id,
        course_id=record.course_id
    ).first()

    if existing_record:
        existing_record.progress = record.progress
    else:
        new_record = LearningRecord(
            user_id=record.user_id,
            course_id=record.course_id,
            progress=record.progress
        )
        db.add(new_record)

    db.commit()
    return {"record_id": existing_record.id if existing_record else new_record.id}


@router.get("/users/{user_id}/recommendations")
def get_recommendations(user_id: int, db: Session = Depends(get_db)):
    user = db.query(User).get(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    records = db.query(LearningRecord).filter_by(user_id=user_id).all()

    if not records:
        return get_popular_courses(db)

    return recommend_courses(user_id, records, db)


# ====================
# 推荐系统优化实现
# ====================
def recommend_courses(user_id: int, records: List[LearningRecord], db: Session) -> Dict:
    """
    基于协同过滤的推荐算法
    1. 找到相似用户
    2. 收集候选课程
    3. 过滤已学课程
    4. 按热度排序
    """
    similar_users = get_similar_users(user_id, records, db)
    return process_recommendations(similar_users, records, db)


def get_similar_users(user_id: int, records: List[LearningRecord], db: Session) -> List[int]:
    """
    使用稀疏矩阵优化相似用户计算
    """
    target_courses = {r.course_id for r in records}

    # 构建稀疏矩阵
    user_matrix = lil_matrix((db.query(User).count(), len(target_courses)), dtype=float)
    course_map = {cid: i for i, cid in enumerate(target_courses)}

    for record in db.query(LearningRecord).filter(LearningRecord.course_id.in_(target_courses)):
        col = course_map.get(record.course_id)
        if col is not None:
            user_matrix[record.user_id - 1, col] = record.progress

    target_vector = user_matrix[user_id - 1].toarray().flatten()

    if target_vector.sum() == 0:
        return []

    similarities = cosine_similarity([target_vector], user_matrix.toarray()).flatten()

    # 排除当前用户
    similar_users = [
        (i + 1, sim) for i, sim in enumerate(similarities)
        if (i + 1) != user_id and sim > 0.2  # 相似度阈值过滤
    ]

    # 按相似度排序取前3
    similar_users.sort(key=lambda x: x[1], reverse=True)
    return [uid for uid, _ in similar_users[:3]]


def process_recommendations(similar_users: List[int], records: List[LearningRecord], db: Session) -> Dict:
    """
    处理推荐结果：
    - 收集候选课程
    - 过滤已学课程
    - 计算课程热度
    """
    learned_courses = {r.course_id for r in records}
    candidate_courses = set()

    for uid in similar_users:
        for record in db.query(LearningRecord).filter_by(user_id=uid):
            if record.course_id not in learned_courses:
                candidate_courses.add(record.course_id)

    if not candidate_courses:
        return get_popular_courses(db)

    # 计算课程热度
    course_popularity = {}
    for cid in candidate_courses:
        course_popularity[cid] = db.query(LearningRecord).filter_by(course_id=cid).count()

    # 按热度排序
    sorted_courses = sorted(course_popularity.items(), key=lambda x: (-x[1], x[0]))
    recommended_courses = db.query(Course).filter(Course.id.in_([cid for cid, _ in sorted_courses[:5]])).all()

    return {"recommended_courses": [course.title for course in recommended_courses]}


@cached(popular_courses_cache)
def get_popular_courses(db: Session) -> Dict:
    """获取热门课程（带缓存）"""
    popular_courses = (
        db.query(
            Course.id,
            func.count(LearningRecord.id).label("course_count")
        )
        .join(LearningRecord, Course.id == LearningRecord.course_id)
        .group_by(Course.id)
        .order_by(func.count(LearningRecord.id).desc())  # 补充括号
        .limit(5)
        .all()
    )

    return {"recommended_courses": [
        db.query(Course).get(course[0]).title
        for course in popular_courses
    ]}


# ====================
# 注册路由
# ====================
app.include_router(router)