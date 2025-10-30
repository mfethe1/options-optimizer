"""
Position Management API Routes
Handles manual entry, CSV import/export, and position enrichment
"""
from fastapi import APIRouter, HTTPException, UploadFile, File, Query
from fastapi.responses import StreamingResponse
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from datetime import datetime
import io
import logging

from ..data.position_manager import PositionManager
from ..data.csv_position_service import CSVPositionService
from ..data.position_enrichment_service import PositionEnrichmentService

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/positions", tags=["positions"])

# Initialize services
position_manager = PositionManager()
csv_service = CSVPositionService(position_manager)
enrichment_service = PositionEnrichmentService(position_manager)


# Pydantic models for API
class StockPositionCreate(BaseModel):
    symbol: str = Field(..., description="Stock ticker symbol")
    quantity: int = Field(..., gt=0, description="Number of shares")
    entry_price: float = Field(..., gt=0, description="Entry price per share")
    entry_date: Optional[str] = Field(None, description="Entry date (YYYY-MM-DD)")
    target_price: Optional[float] = Field(None, gt=0, description="Target price")
    stop_loss: Optional[float] = Field(None, gt=0, description="Stop loss price")
    notes: Optional[str] = Field(None, description="Position notes")


class OptionPositionCreate(BaseModel):
    symbol: str = Field(..., description="Underlying ticker symbol")
    option_type: str = Field(..., description="Option type: 'call' or 'put'")
    strike: float = Field(..., gt=0, description="Strike price")
    expiration_date: str = Field(..., description="Expiration date (YYYY-MM-DD)")
    quantity: int = Field(..., gt=0, description="Number of contracts")
    premium_paid: float = Field(..., gt=0, description="Premium paid per contract")
    entry_date: Optional[str] = Field(None, description="Entry date (YYYY-MM-DD)")
    target_price: Optional[float] = Field(None, gt=0, description="Target option price")
    target_profit_pct: Optional[float] = Field(None, gt=0, description="Target profit %")
    stop_loss_pct: Optional[float] = Field(None, gt=0, description="Stop loss %")
    notes: Optional[str] = Field(None, description="Position notes")


class PositionUpdate(BaseModel):
    target_price: Optional[float] = None
    stop_loss: Optional[float] = None
    target_profit_pct: Optional[float] = None
    stop_loss_pct: Optional[float] = None
    notes: Optional[str] = None


class ChaseConversionStats(BaseModel):
    """Statistics from Chase CSV conversion"""
    total_rows: int
    options_converted: int
    cash_skipped: int
    conversion_errors: int
    error_details: List[str]


class CSVImportResponse(BaseModel):
    success: int
    failed: int
    errors: List[str]
    position_ids: List[str]
    chase_conversion: Optional[ChaseConversionStats] = None


# Stock Position Endpoints
@router.post("/stocks", response_model=Dict[str, Any])
async def create_stock_position(position: StockPositionCreate):
    """Create a new stock position"""
    try:
        position_id = position_manager.add_stock_position(
            symbol=position.symbol.upper(),
            quantity=position.quantity,
            entry_price=position.entry_price,
            entry_date=position.entry_date,
            target_price=position.target_price,
            stop_loss=position.stop_loss,
            notes=position.notes
        )
        
        # Enrich the position
        pos = position_manager.get_stock_position(position_id)
        enrichment_service.enrich_stock_position(pos)
        position_manager.save_positions()
        
        return {
            "position_id": position_id,
            "message": "Stock position created successfully",
            "position": pos.to_dict()
        }
    except Exception as e:
        logger.error(f"Error creating stock position: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stocks", response_model=List[Dict[str, Any]])
async def get_stock_positions(enrich: bool = Query(False, description="Enrich with real-time data")):
    """Get all stock positions"""
    try:
        positions = position_manager.get_all_stock_positions()
        
        if enrich:
            for pos in positions:
                enrichment_service.enrich_stock_position(pos)
            position_manager.save_positions()
        
        return [pos.to_dict() for pos in positions]
    except Exception as e:
        logger.error(f"Error getting stock positions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stocks/{position_id}", response_model=Dict[str, Any])
async def get_stock_position(position_id: str, enrich: bool = Query(False)):
    """Get a specific stock position"""
    try:
        position = position_manager.get_stock_position(position_id)
        if not position:
            raise HTTPException(status_code=404, detail="Position not found")
        
        if enrich:
            enrichment_service.enrich_stock_position(position)
            position_manager.save_positions()
        
        return position.to_dict()
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting stock position: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.patch("/stocks/{position_id}", response_model=Dict[str, Any])
async def update_stock_position(position_id: str, update: PositionUpdate):
    """Update a stock position"""
    try:
        updates = {k: v for k, v in update.dict().items() if v is not None}
        success = position_manager.update_stock_position(position_id, **updates)
        
        if not success:
            raise HTTPException(status_code=404, detail="Position not found")
        
        position = position_manager.get_stock_position(position_id)
        return {
            "message": "Position updated successfully",
            "position": position.to_dict()
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating stock position: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/stocks/{position_id}")
async def delete_stock_position(position_id: str):
    """Delete a stock position"""
    try:
        success = position_manager.remove_stock_position(position_id)
        if not success:
            raise HTTPException(status_code=404, detail="Position not found")
        
        return {"message": "Position deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting stock position: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Option Position Endpoints
@router.post("/options", response_model=Dict[str, Any])
async def create_option_position(position: OptionPositionCreate):
    """Create a new option position"""
    try:
        if position.option_type.lower() not in ['call', 'put']:
            raise HTTPException(status_code=400, detail="Option type must be 'call' or 'put'")
        
        position_id = position_manager.add_option_position(
            symbol=position.symbol.upper(),
            option_type=position.option_type.lower(),
            strike=position.strike,
            expiration_date=position.expiration_date,
            quantity=position.quantity,
            premium_paid=position.premium_paid,
            entry_date=position.entry_date,
            target_price=position.target_price,
            target_profit_pct=position.target_profit_pct,
            stop_loss_pct=position.stop_loss_pct,
            notes=position.notes
        )
        
        # Enrich the position
        pos = position_manager.get_option_position(position_id)
        enrichment_service.enrich_option_position(pos)
        position_manager.save_positions()
        
        return {
            "position_id": position_id,
            "message": "Option position created successfully",
            "position": pos.to_dict()
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating option position: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/options", response_model=List[Dict[str, Any]])
async def get_option_positions(enrich: bool = Query(False, description="Enrich with real-time data")):
    """Get all option positions"""
    try:
        positions = position_manager.get_all_option_positions()
        
        if enrich:
            for pos in positions:
                enrichment_service.enrich_option_position(pos)
            position_manager.save_positions()
        
        return [pos.to_dict() for pos in positions]
    except Exception as e:
        logger.error(f"Error getting option positions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/options/{position_id}", response_model=Dict[str, Any])
async def get_option_position(position_id: str, enrich: bool = Query(False)):
    """Get a specific option position"""
    try:
        position = position_manager.get_option_position(position_id)
        if not position:
            raise HTTPException(status_code=404, detail="Position not found")
        
        if enrich:
            enrichment_service.enrich_option_position(position)
            position_manager.save_positions()
        
        return position.to_dict()
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting option position: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.patch("/options/{position_id}", response_model=Dict[str, Any])
async def update_option_position(position_id: str, update: PositionUpdate):
    """Update an option position"""
    try:
        updates = {k: v for k, v in update.dict().items() if v is not None}
        success = position_manager.update_option_position(position_id, **updates)
        
        if not success:
            raise HTTPException(status_code=404, detail="Position not found")
        
        position = position_manager.get_option_position(position_id)
        return {
            "message": "Position updated successfully",
            "position": position.to_dict()
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating option position: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/options/{position_id}")
async def delete_option_position(position_id: str):
    """Delete an option position"""
    try:
        success = position_manager.remove_option_position(position_id)
        if not success:
            raise HTTPException(status_code=404, detail="Position not found")
        
        return {"message": "Position deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting option position: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# CSV Template Endpoints
@router.get("/templates/stocks")
async def download_stock_template():
    """Download CSV template for stock positions"""
    try:
        csv_content = csv_service.generate_stock_template()
        
        return StreamingResponse(
            io.StringIO(csv_content),
            media_type="text/csv",
            headers={
                "Content-Disposition": f"attachment; filename=stock_positions_template_{datetime.now().strftime('%Y%m%d')}.csv"
            }
        )
    except Exception as e:
        logger.error(f"Error generating stock template: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/templates/options")
async def download_option_template():
    """Download CSV template for option positions"""
    try:
        csv_content = csv_service.generate_option_template()

        return StreamingResponse(
            io.StringIO(csv_content),
            media_type="text/csv",
            headers={
                "Content-Disposition": f"attachment; filename=option_positions_template_{datetime.now().strftime('%Y%m%d')}.csv"
            }
        )
    except Exception as e:
        logger.error(f"Error generating option template: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# CSV Export Endpoints
@router.get("/export/stocks")
async def export_stock_positions():
    """Export current stock positions to CSV"""
    try:
        csv_content = csv_service.export_stock_positions()

        return StreamingResponse(
            io.StringIO(csv_content),
            media_type="text/csv",
            headers={
                "Content-Disposition": f"attachment; filename=stock_positions_{datetime.now().strftime('%Y%m%d')}.csv"
            }
        )
    except Exception as e:
        logger.error(f"Error exporting stock positions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/export/options")
async def export_option_positions():
    """Export current option positions to CSV"""
    try:
        csv_content = csv_service.export_option_positions()

        return StreamingResponse(
            io.StringIO(csv_content),
            media_type="text/csv",
            headers={
                "Content-Disposition": f"attachment; filename=option_positions_{datetime.now().strftime('%Y%m%d')}.csv"
            }
        )
    except Exception as e:
        logger.error(f"Error exporting option positions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# CSV Import Endpoints
@router.post("/import/stocks", response_model=CSVImportResponse)
async def import_stock_positions(
    file: UploadFile = File(...),
    replace_existing: bool = Query(False, description="Replace all existing stock positions")
):
    """Import stock positions from CSV file"""
    try:
        # Read file content
        content = await file.read()
        csv_content = content.decode('utf-8')

        # Import positions
        results = csv_service.import_stock_positions(csv_content, replace_existing)

        # Enrich imported positions
        if results['success'] > 0:
            enrichment_service.enrich_all_positions()

        return CSVImportResponse(**results)
    except Exception as e:
        logger.error(f"Error importing stock positions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/import/options", response_model=CSVImportResponse)
async def import_option_positions(
    file: UploadFile = File(...),
    replace_existing: bool = Query(False, description="Replace all existing option positions"),
    chase_format: bool = Query(False, description="Is this a Chase.com CSV export? (will auto-convert)")
):
    """
    Import option positions from CSV file

    Supports two formats:
    1. Standard format (our CSV template)
    2. Chase.com export format (set chase_format=true)

    When chase_format=true, the system will automatically:
    - Parse Chase's complex CSV structure
    - Extract option details from description fields
    - Convert dates to our format
    - Preserve Chase's pricing data for validation
    - Skip cash positions
    """
    try:
        # Read file content
        content = await file.read()
        csv_content = content.decode('utf-8')

        # Import positions (with optional Chase conversion)
        results = csv_service.import_option_positions(
            csv_content,
            replace_existing,
            chase_format=chase_format
        )

        # Enrich imported positions
        if results['success'] > 0:
            enrichment_service.enrich_all_positions()

        return CSVImportResponse(**results)
    except Exception as e:
        logger.error(f"Error importing option positions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Portfolio Summary Endpoint
@router.get("/summary", response_model=Dict[str, Any])
async def get_portfolio_summary(enrich: bool = Query(True, description="Enrich with real-time data")):
    """Get portfolio summary with enriched data"""
    try:
        if enrich:
            summary = enrichment_service.get_enriched_portfolio_summary()
        else:
            summary = position_manager.get_portfolio_summary()

        return summary
    except Exception as e:
        logger.error(f"Error getting portfolio summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Bulk Enrichment Endpoint
@router.post("/enrich", response_model=Dict[str, Any])
async def enrich_all_positions():
    """Enrich all positions with real-time data and Greeks"""
    try:
        results = enrichment_service.enrich_all_positions()
        return {
            "message": "Positions enriched successfully",
            "results": results
        }
    except Exception as e:
        logger.error(f"Error enriching positions: {e}")
        raise HTTPException(status_code=500, detail=str(e))

