"""empty message

Revision ID: ecd91663497b
Revises: 
Create Date: 2024-04-08 19:37:12.076392

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'ecd91663497b'
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.create_table('User',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('email', sa.String(length=80), nullable=False),
    sa.Column('joined_on', sa.DateTime(), nullable=False),
    sa.Column('is_admin', sa.Boolean(), nullable=True),
    sa.Column('is_active', sa.Boolean(), nullable=True),
    sa.Column('password', sa.String(length=120), nullable=False),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('email')
    )
    op.create_table('csv_data',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('user_id', sa.Integer(), nullable=False),
    sa.Column('filename', sa.String(length=255), nullable=True),
    sa.Column('time', sa.String(length=255), nullable=True),
    sa.Column('accv', sa.Float(), nullable=True),
    sa.Column('accml', sa.Float(), nullable=True),
    sa.Column('accap', sa.Float(), nullable=True),
    sa.Column('visit_x', sa.String(length=255), nullable=True),
    sa.Column('age', sa.Integer(), nullable=True),
    sa.Column('sex', sa.String(length=10), nullable=True),
    sa.Column('years_since_dx', sa.Integer(), nullable=True),
    sa.Column('updrsiii_on', sa.Float(), nullable=True),
    sa.Column('updrsiii_off', sa.Float(), nullable=True),
    sa.Column('nfogq', sa.Integer(), nullable=True),
    sa.Column('medication', sa.String(length=255), nullable=True),
    sa.Column('init', sa.String(length=255), nullable=True),
    sa.Column('completion', sa.String(length=255), nullable=True),
    sa.Column('kinetic', sa.String(length=255), nullable=True),
    sa.ForeignKeyConstraint(['user_id'], ['User.id'], ),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('id')
    )
    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_table('csv_data')
    op.drop_table('User')
    # ### end Alembic commands ###