from fastapi import APIRouter
from v1.v2 import endpoints_router

router = APIRouter()

router.include_router(endpoints_router,prefix="")
