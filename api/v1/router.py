from fastapi import APIRouter
from api.v1.v2.endpoints import router as v2_router

router = APIRouter()
router.include_router(v2_router, prefix="/v2")